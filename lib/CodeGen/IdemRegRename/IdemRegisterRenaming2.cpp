//===----- IdemRegisterRenaming.cpp - Register Renaming after RA ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-renaming"


#include <llvm/CodeGen/MachineIdempotentRegions.h>
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <llvm/ADT/SetOperations.h>
#include "llvm/Target/TargetData.h"
#include <llvm/PassSupport.h>

#include "IdemUtil.h"
#include "IdempotentRegionLiveInsGather.h"
#include "LiveIntervalAnalysisIdem.h"

#include <queue>
#include <utility>
using namespace llvm;

/// @author Jianping Zeng.
namespace {

struct MIOp {
  MachineBasicBlock::iterator mi;
  unsigned index;

  MachineOperand &operator->() {
    return mi->getOperand(index);
  }
  const MachineOperand &operator->() const {
    return mi->getOperand(index);
  }

  MIOp(MachineInstr *MI, unsigned Index) : mi(MI), index(Index) {}

  bool operator==(MIOp &rhs) {
    return &*mi == &*rhs.mi && index == rhs.index;
  }

  bool operator==(const MIOp &rhs) const {
    return &*mi == &*rhs.mi && index == rhs.index;
  }

  MIOp &operator=(const MIOp &rhs) = default;

  bool operator !=(MIOp &rhs) {
    return !(*this == rhs);
  }

  bool operator !=(const MIOp &rhs) const {
    return !(*this == rhs);
  }
};

struct AntiDeps {
  unsigned reg;
  std::vector<MIOp> uses;
  std::vector<MIOp> defs;

  AntiDeps() = default;

  AntiDeps(unsigned Reg, std::vector<MIOp> &Uses,
           std::vector<MIOp> &Defs)
      : reg(Reg), uses(), defs() {
    uses.insert(uses.end(), Uses.begin(), Uses.end());
    defs.insert(defs.end(), Defs.begin(), Defs.end());
  }

  bool operator==(const AntiDeps &rhs) const {
    return reg == rhs.reg && uses == rhs.uses && defs == rhs.defs;
  }

  bool operator==(AntiDeps &rhs) {
    return reg == rhs.reg && uses == rhs.uses && defs == rhs.defs;
  }
};

class IdemRegisterRenamer : public MachineFunctionPass {
public:
  static char ID;
  IdemRegisterRenamer() : MachineFunctionPass(ID) {
    initializeIdemRegisterRenamerPass(*PassRegistry::getPassRegistry());
    tii = nullptr;
    tri = nullptr;
    mir = nullptr;
    gather = nullptr;
    li = nullptr;
    mf = nullptr;
    mri = nullptr;
    mfi = nullptr;
    dt = nullptr;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalAnalysisIdem>();
    AU.addRequired<MachineIdempotentRegions>();
    AU.addRequired<MachineDominatorTree>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const override {
    return "Register Renaming for Idempotence pass";
  }

  void clear() {
    if (gather)
      delete gather;

    tii = nullptr;
    tri = nullptr;
    li = nullptr;
    mf = nullptr;
    mri = nullptr;
    mfi = nullptr;
    antiDeps.clear();
  }

private:
  inline void collectLiveInRegistersForRegions();
  void computeAntiDependenceSet();
  void gatherAntiDeps(MachineInstr *idem);
  bool handleAntiDependences(bool &needRecompute);
  void collectAntiDepsTrace(unsigned,
                            const MachineBasicBlock::iterator &,
                            const MachineBasicBlock::iterator &,
                            MachineBasicBlock *,
                            std::vector<MIOp>,
                            std::vector<MIOp>,
                            std::set<MachineBasicBlock*> &visited);
  void useDefChainEnds(unsigned reg,
                       std::set<MachineBasicBlock*> &visited,
                       MachineBasicBlock::iterator start,
                       MachineBasicBlock::iterator end,
                       MachineBasicBlock *mbb, bool &ends);

  bool isTwoAddressInstr(MachineInstr *useMI, unsigned reg);

  void spillCurrentUse(AntiDeps &ad);

  unsigned choosePhysRegForRenaming(MachineOperand *use,
                                    LiveIntervalIdem *interval,
                                    DenseSet<unsigned> &unallocableRegs);
  void filterUnavailableRegs(MachineOperand *use,
                             BitVector &allocSet,
                             bool allowsAntiDep);

  bool legalToReplace(unsigned newReg, unsigned oldReg);
  unsigned tryChooseFreeRegister(LiveIntervalIdem &interval,
                                 int useReg,
                                 BitVector &allocSet);
  unsigned tryChooseBlockedRegister(LiveIntervalIdem &interval,
                                    int useReg,
                                    BitVector &allocSet);
  void getSpilledSubLiveInterval(LiveIntervalIdem *interval,
                                 std::vector<LiveIntervalIdem *> &spilledItrs);
  void getAllocableRegs(unsigned useReg, std::set<unsigned> &allocables);

  void revisitSpilledInterval(std::set<unsigned> &allocables, std::vector<LiveIntervalIdem *> &spilled);

  void processHandledIntervals(std::vector<LiveIntervalIdem *> &handled,
                               unsigned currentStart);
  void insertSpillingCodeForInterval(LiveIntervalIdem *spilledItr);

  void assignRegOrStackSlotAtInterval(std::set<unsigned> &allocables,
                                      LiveIntervalIdem *interval,
                                      std::vector<LiveIntervalIdem *> &handled,
                                      std::vector<LiveIntervalIdem *> &spilled);

  void getUsesSetOfDef(MachineOperand *def,
                       std::vector<MIOp> &usesAndDefs,
                       bool &canReplace);

  unsigned getFreeRegisterForRenaming(unsigned useReg,
                                      LiveIntervalIdem *interval,
                                      DenseSet<unsigned> unallocableRegs);

  void walkDFSToGatheringUses(unsigned reg,
                              MachineBasicBlock::iterator begin,
                              MachineBasicBlock::iterator end,
                              MachineBasicBlock *mbb,
                              std::set<MachineBasicBlock *> &visited,
                              std::vector<MIOp> &usesAndDefs,
                              bool &canReplace,
                              bool seeIdem);

  void collectUnallocableRegs(MachineBasicBlock::reverse_iterator begin,
                              MachineBasicBlock::reverse_iterator end,
                              MachineBasicBlock *mbb,
                              std::set<MachineBasicBlock*> &visited,
                              DenseSet<unsigned> &unallocableRegs);

  unsigned getNumFreeRegs(unsigned reg, DenseSet<unsigned> &unallocableRegs);

  bool shouldSpillCurrent(AntiDeps &ad,
                          DenseSet<unsigned> &unallocableRegs,
                          std::vector<IdempotentRegion *> &regions);

  void willRenameCauseOtherAntiDep(MachineBasicBlock::iterator begin,
                                   MachineBasicBlock::iterator end, MachineBasicBlock *mbb,
                                   unsigned reg, std::set<MachineBasicBlock *> &visited,
                                   bool &canRename);

  void updateLiveInOfPriorRegions(MachineBasicBlock::reverse_iterator begin,
                                  MachineBasicBlock::reverse_iterator end,
                                  MachineBasicBlock *mbb,
                                  std::set<MachineBasicBlock*> &visited,
                                  unsigned useReg,
                                  bool &seenRedef);

  bool partialEquals(unsigned reg1, unsigned reg2) {
    assert(TargetRegisterInfo::isPhysicalRegister(reg1) &&
        TargetRegisterInfo::isPhysicalRegister(reg2));

    if (reg1 == reg2)
      return true;
    for (const unsigned *r = tri->getSubRegisters(reg1); *r; ++r)
      if (*r == reg2)
        return true;

    for (const unsigned *r = tri->getSubRegisters(reg2); *r; ++r)
      if (*r == reg1)
        return true;

    return false;
  }

  bool willRaiseAntiDep(unsigned useReg,
                        MachineBasicBlock::iterator begin,
                        MachineBasicBlock::iterator end,
                        MachineBasicBlock *mbb/*,
                        std::set<MachineBasicBlock *> &visited*/);

  bool willRaiseAntiDepsInLoop(unsigned reg,
                               const MachineBasicBlock::iterator &idem,
                               const MachineBasicBlock::iterator &end,
                               MachineBasicBlock *mbb) {
    if (!mbb) return false;

    std::set<MachineBasicBlock*> visited;
    std::vector<MachineBasicBlock*> worklist;
    worklist.push_back(mbb);
    while (!worklist.empty()) {
      auto cur = worklist.back();
      worklist.pop_back();

      if (!visited.insert(cur).second)
        continue;

      if (!cur->empty()) {
        for (auto &mi : *cur) {
          if (tii->isIdemBoundary(&mi))
            return true;
        }
      }
      for (auto succ = cur->succ_begin(), end = cur->succ_end(); succ != end; ++succ)
        if (!visited.count(*succ))
          worklist.push_back(*succ);
    }

    return false;
  }

  void computeAntiDepsInLoop(unsigned reg,
                             const MachineBasicBlock::iterator &idem,
                             const MachineBasicBlock::iterator &end,
                             MachineBasicBlock *mbb,
                             std::vector<MIOp> uses,
                             std::vector<MIOp> defs) {
    for (auto itr = idem; itr != end; ++itr) {
      if (tii->isIdemBoundary(itr))
        return;

      // iterate over operands from right to left, which means cope with
      // use oprs firstly, then def regs.
      for (int i = itr->getNumOperands() - 1; i >= 0; i--) {
        auto mo = itr->getOperand(i);
        if (!mo.isReg() || !mo.getReg() || mo.getReg() != reg)
          continue;

        if (mo.isUse())
          uses.emplace_back(itr, i);
        else {
          defs.emplace_back(itr, i);

          ++itr;
          bool ends = false;
          std::set<MachineBasicBlock *> visited;
          useDefChainEnds(reg, visited, itr, end, mbb, ends);
          if (ends) {
            // Construct anti-dependencies according uses and defs set.
            antiDeps.emplace_back(reg, uses, defs);
            return;
          }
          --itr;
        }
      }
    }

    if (!mbb->succ_empty()) {
      for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ) {
        // Avoiding cycle walking over CFG.
        // If the next block is the loop header block and it have idem instr, we have to visit it.
        if (!dt->dominates(*succ, mbb))
          computeAntiDepsInLoop(reg, (*succ)->begin(), (*succ)->end(), *succ, uses, defs);
      }
    }
  }

  void addRegisterWithSubregs(DenseSet<unsigned> &set, unsigned reg) {
    set.insert(reg);
    if (!TargetRegisterInfo::isStackSlot(reg) &&
        TargetRegisterInfo::isPhysicalRegister(reg)) {
      for (const unsigned *r = tri->getSubRegisters(reg); *r; ++r)
        set.insert(*r);
    }
  }

private:
  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;
  MachineIdempotentRegions *mir;
  std::vector<AntiDeps> antiDeps;
  LiveInsGather *gather;
  LiveIntervalAnalysisIdem *li;
  MachineFunction *mf;
  MachineRegisterInfo *mri;
  MachineFrameInfo *mfi;
  MachineDominatorTree *dt;
};
}

INITIALIZE_PASS_BEGIN(IdemRegisterRenamer, "reg-renaming",
                      "Register Renaming for Idempotence", false, false)
  INITIALIZE_PASS_DEPENDENCY(LiveIntervalAnalysisIdem)
  INITIALIZE_PASS_DEPENDENCY(MachineIdempotentRegions)
  INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(IdemRegisterRenamer, "reg-renaming",
                    "Register Renaming for Idempotence", false, false)

char IdemRegisterRenamer::ID = 0;

FunctionPass *llvm::createIdemRegisterRenamerPass() {
  return new IdemRegisterRenamer();
}

void IdemRegisterRenamer::collectLiveInRegistersForRegions() {
  gather->run();
}

LLVM_ATTRIBUTE_UNUSED static bool contains(std::vector<MIOp> &set, unsigned reg) {
  for (auto &mo : set)
    if (mo.mi->getOperand(mo.index).getReg() == reg)
      return true;
  return false;
}

void IdemRegisterRenamer::useDefChainEnds(unsigned reg,
                                          std::set<MachineBasicBlock*> &visited,
                                          MachineBasicBlock::iterator start,
                                          MachineBasicBlock::iterator end,
                                          MachineBasicBlock *mbb,
                                          bool &ends) {
  ends = true;
  if (!visited.insert(mbb).second)
    return;

  if (start != end) {
    for (; start != end; ++start) {
      MachineInstr* mi = start;
      if (tii->isIdemBoundary(mi))
        return;

      std::vector<MachineOperand*> defs, uses;

      for (int i = mi->getNumOperands() - 1; i >= 0; i--) {
        auto mo = mi->getOperand(i);
        if (!mo.isReg() || !mo.getReg() || mo.getReg() != reg)
          continue;
        if (mo.isDef()) defs.push_back(&mo);
        else uses.push_back(&mo);
      }

      if (!defs.empty() || !uses.empty()) {
        ends = !uses.empty();
        return;
      }
    }
  }
  std::for_each(mbb->succ_begin(), mbb->succ_end(), [&](MachineBasicBlock *succ) {
    useDefChainEnds(reg, visited, succ->begin(), succ->end(), succ, ends);
  });
}

void IdemRegisterRenamer::collectAntiDepsTrace(unsigned reg,
                                               const MachineBasicBlock::iterator &idem,
                                               const MachineBasicBlock::iterator &end,
                                               MachineBasicBlock *mbb,
                                               std::vector<MIOp> uses,
                                               std::vector<MIOp> defs,
                                               std::set<MachineBasicBlock*> &visited) {
  /*if (mbb->getParent()->getFunction()->getName() == "predictor_zero" &&
  mbb->getName() == "for.end") {
    llvm::errs() << mbb->getParent()->getFunction()->getName() << "\n";
    llvm::errs() << mbb->getName() << "\n\n\n";
    mf->dump();
  }*/

  if (!mbb) return;
  visited.insert(mbb);

  for (auto itr = idem; itr != end; ++itr) {
    if (tii->isIdemBoundary(itr))
      return;

    // iterate over operands from right to left, which means cope with
    // use oprs firstly, then def regs.
    for (int i = itr->getNumOperands() - 1; i >= 0; i--) {
      auto mo = itr->getOperand(i);
      if (!mo.isReg() || !mo.getReg() || mo.getReg() != reg)
        continue;

      if (mo.isUse())
        uses.emplace_back(itr, i);
      else {
        defs.emplace_back(itr, i);

        ++itr;
        bool ends = false;
        std::set<MachineBasicBlock *> tmpVisited;
        useDefChainEnds(reg, tmpVisited, itr, end, mbb, ends);
        if (ends) {
          // Construct anti-dependencies according uses and defs set.
          antiDeps.emplace_back(reg, uses, defs);
          return;
        }
        --itr;
      }
    }
  }

  if (!mbb->succ_empty()) {
    for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ) {
      /*if (mbb->getParent()->getFunction()->getName() == "predictor_zero" &&
          mbb->getName() == "for.end") {
        llvm::errs() << (*succ)->getName() << "\n\n\n";
      }*/

      // Avoiding cycle walking over CFG.
      // If the next block is the loop header block and it have idem instr, we have to visit it.
      if (!visited.count(*succ))
        collectAntiDepsTrace(reg, (*succ)->begin(), (*succ)->end(), *succ, uses, defs, visited);
      else
        computeAntiDepsInLoop(reg, (*succ)->begin(), (*succ)->end(), *succ, uses, defs);
    }
  }
}

void IdemRegisterRenamer::gatherAntiDeps(MachineInstr *idem) {
  auto liveIns = gather->getIdemLiveIns(idem);
  if (liveIns.empty())
    return;

  auto begin = ++MachineBasicBlock::iterator(idem);
  for (auto reg : liveIns) {
  /*if (idem->getParent()->getName() == "if.end35"*//* && reg == 61*//*) {
      llvm::errs()<<"LiveIns: [";
      for (auto reg : liveIns)
        llvm::errs()<<tri->getName(reg)<<", ";
      llvm::errs()<<"]\n";
    }*/

    std::set<MachineBasicBlock*> visited;
    // for an iteration of each live-in register, renew the visited set.
    collectAntiDepsTrace(reg, begin, idem->getParent()->end(),
                         idem->getParent(),
                         std::vector<MIOp>(),
                         std::vector<MIOp>(),
                         visited);
  }
}

void IdemRegisterRenamer::computeAntiDependenceSet() {
  for (auto &itr : *mir) {
    MachineInstr *idem = &itr->getEntry();
    assert(idem && tii->isIdemBoundary(idem));
    gatherAntiDeps(idem);
  }
}

bool IdemRegisterRenamer::isTwoAddressInstr(MachineInstr *useMI, unsigned reg) {
  // We should not rename the two-address instruction.
  auto mcID = useMI->getDesc();
  unsigned numOps = useMI->isInlineAsm() ? useMI->getNumOperands() : mcID.getNumOperands();
  for (unsigned i = 0; i < numOps; i++) {
    unsigned destIdx;
    if (!useMI->isRegTiedToDefOperand(i, &destIdx))
      continue;

    return useMI->getOperand(destIdx).getReg() == reg;
  }
  return false;
}

static void getDefUses(MachineInstr *mi,
                       std::set<MachineOperand *> *defs,
                       std::set<MachineOperand *> *uses,
                       const BitVector &allocaSets,
                       std::set<unsigned> *defRegs = 0) {
  if (!mi)
    return;

  for (unsigned i = 0, e = mi->getNumOperands(); i < e; i++) {
    MachineOperand *mo = &mi->getOperand(i);
    if (!mo || !mo->isReg() ||
        !mo->getReg() || mo->isImplicit() ||
        !allocaSets.test(mo->getReg()))
      continue;

    unsigned &&reg = mo->getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(reg));

    if (mo->isDef()) {
      if (defs)
        defs->insert(mo);
      if (defRegs)
        defRegs->insert(mo->getReg());
    } else if (mo->isUse() && uses)
      uses->insert(mo);
  }
}
/**
 * Checks if it is legal to replace the oldReg with newReg. If so, return true.
 * @param newReg
 * @param oldReg
 * @return
 */
bool IdemRegisterRenamer::legalToReplace(unsigned newReg, unsigned oldReg) {
  for (unsigned i = 0, e = tri->getNumRegClasses(); i < e; i++) {
    auto rc = tri->getRegClass(i);
    if (tri->canUsedToReplace(newReg) && rc->contains(newReg) && rc->contains(oldReg))
      return true;
  }
  return false;
}

LLVM_ATTRIBUTE_UNUSED void IdemRegisterRenamer::filterUnavailableRegs(MachineOperand *use,
                                                BitVector &allocSet,
                                                bool allowsAntiDep) {
  // remove the defined register by use mi from allocable set.
  std::set<MachineOperand *> defs;
  getDefUses(use->getParent(), &defs, 0, tri->getAllocatableSet(*mf));
  for (MachineOperand *phy : defs)
    allocSet[phy->getReg()] = false;

  // also, we must make sure no the physical register same as
  // use will be assigned.
  allocSet[use->getReg()] = false;

  // Remove some physical register whose register class is not compatible with rc.
  // const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(use->getReg());
  for (int physReg = allocSet.find_first(); physReg != -1; physReg = allocSet.find_next(physReg))
    if (!legalToReplace(physReg, use->getReg()))
      allocSet[physReg] = false;

  // When we use another new register to replace the used register,
  // we may introduce new anti-dependence between new reg and defined register
  // by successive instr.
  //
  // So we need a flag tell us whether anti-dependences occur is allowed or not.
  // if it is allowed, so we should re-handle it after.
  //
  // Before:
  // R0 = R0 + R1
  //    ...
  // R2 = ...
  //
  // After:
  // R2 = R0
  //---------
  // R0 = R2 + R1
  //    ...
  // R2 = ...
  // So anti-dependence occurs again !!!
  if (!allowsAntiDep) {
    typedef MachineBasicBlock::iterator MIItr;
    std::vector<std::pair<MIItr, MIItr>> worklist;
    worklist.emplace_back(use->getParent(), use->getParent()->getParent()->end());
    std::set<MachineBasicBlock *> visited;

    while (!worklist.empty()) {
      auto itrRange = worklist.back();
      worklist.pop_back();

      if (itrRange.first == itrRange.second)
        continue;

      // Also the assigned register can not is same as the defined reg by successive instr.
      auto mbb = itrRange.first->getParent();
      if (!visited.insert(mbb).second)
        continue;

      auto begin = itrRange.first, end = itrRange.second;
      for (++begin; begin != end && !tii->isIdemBoundary(begin); ++begin) {
        std::set<MachineOperand *> defs;
        getDefUses(begin, &defs, 0, allocSet);
        for (auto defMO : defs)
          allocSet[defMO->getReg()] = false;
      }

      if (begin != end)
        return;

      std::for_each(mbb->succ_begin(), mbb->succ_end(), [&](MachineBasicBlock *succ) {
        worklist.emplace_back(succ->begin(), succ->end());
      });
    }
  }
}

unsigned IdemRegisterRenamer::tryChooseFreeRegister(LiveIntervalIdem &interval,
                                                    int useReg,
                                                    BitVector &allocSet) {
  IDEM_DEBUG(llvm::errs() << "Interval for move instr: ";
                 interval.dump(*const_cast<TargetRegisterInfo *>(tri));
                 llvm::errs() << "\n";);

  for (int physReg = allocSet.find_first(); physReg > 0; physReg = allocSet.find_next(physReg)) {
    if (li->intervals.count(physReg)) {
      LiveIntervalIdem *itrv = li->intervals[physReg];

      IDEM_DEBUG(llvm::errs() << "Candidate interval: ";
                     itrv->dump(*const_cast<TargetRegisterInfo *>(tri));
                     llvm::errs() << "\n";);

      if (!itrv->intersects(&interval)) {
        // we only consider those live interval which doesn't interfere with current
        // interval.
        // No matching in register class should be ignored.
        // Avoiding LiveIn register(such as argument register).
        if (!legalToReplace(physReg, useReg))
          continue;

        return physReg;
      }
    } else if (legalToReplace(physReg, useReg)) {
      // current physReg is free, so return it.
      return static_cast<unsigned int>(physReg);
    }
  }
  return 0;
}

void IdemRegisterRenamer::getSpilledSubLiveInterval(LiveIntervalIdem *interval,
                                                    std::vector<LiveIntervalIdem *> &spilledItrs) {

  for (auto begin = interval->usepoint_begin(),
           end = interval->usepoint_end(); begin != end; ++begin) {
    LiveIntervalIdem *verifyLI = new LiveIntervalIdem;
    verifyLI->addUsePoint(begin->id, begin->mo);
    unsigned from, to;
    if (begin->mo->isUse()) {
      to = li->mi2Idx[begin->mo->getParent()];
      from = to - 1;
    } else {
      from = li->mi2Idx[begin->mo->getParent()];
      to = from + 1;
    }
    verifyLI->addRange(from, to);
    verifyLI->reg = 0;
    // tells compiler not to evict this spilling interval.
    verifyLI->costToSpill = UINT32_MAX;
  }
}

void IdemRegisterRenamer::getAllocableRegs(unsigned useReg, std::set<unsigned> &allocables) {
  for (unsigned i = 0, e = tri->getNumRegClasses(); i < e; i++) {
    auto rc = tri->getRegClass(i);
    if (rc->contains(useReg))
      for (auto reg : rc->getRawAllocationOrder(*mf))
        allocables.insert(reg);
  }
}

// for a group of sub live interval caused by splitting the original live interval.
// all of sub intervals in the same group have to be assigned a same frame index.
static unsigned groupId = 1;
static std::map<LiveIntervalIdem *, unsigned> intervalGrpId;
static std::map<unsigned, int> grp2FrameIndex;

static unsigned getOrGroupId(std::vector<LiveIntervalIdem *> itrs) {
  if (itrs.empty())
    return 0;
  for (LiveIntervalIdem *itr : itrs) {
    if (intervalGrpId.count(itr))
      return intervalGrpId[itr];

    intervalGrpId[itr] = groupId;
  }
  ++groupId;
  return groupId - 1;
}

static unsigned getOrGroupId(LiveIntervalIdem *itr) {
  assert(itr);
  assert (intervalGrpId.count(itr));
  return intervalGrpId[itr];
}

static bool hasFrameSlot(LiveIntervalIdem *itr) {
  return grp2FrameIndex.count(getOrGroupId(itr));
}

static void setFrameIndex(LiveIntervalIdem *itr, int frameIndex) {
  unsigned grpId = getOrGroupId(itr);
  grp2FrameIndex[grpId] = frameIndex;
}

static int getFrameIndex(LiveIntervalIdem *itr) {
  assert(hasFrameSlot(itr));
  return grp2FrameIndex[getOrGroupId(itr)];
}

LLVM_ATTRIBUTE_UNUSED static MachineInstr *getPrevMI(MachineInstr *mi) {
  if (!mi || !mi->getParent())
    return nullptr;
  return ilist_traits<MachineInstr>::getPrev(mi);
}

static MachineInstr *getNextMI(MachineInstr *mi) {
  if (!mi || !mi->getParent())
    return nullptr;
  return ilist_traits<MachineInstr>::getNext(mi);
}

void IdemRegisterRenamer::insertSpillingCodeForInterval(LiveIntervalIdem *spilledItr) {
  int frameIndex;

  auto interval = spilledItr;
  MachineOperand *mo = interval->usepoint_begin()->mo;
  MachineInstr *mi = mo->getParent();
  assert(mo->isReg());
  unsigned usedReg = interval->reg;
  mo->setReg(usedReg);

  const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(mo->getReg());
  if (hasFrameSlot(spilledItr))
    frameIndex = getFrameIndex(spilledItr);
  else {
    frameIndex = mfi->CreateSpillStackObject(rc->getSize(), rc->getAlignment());
    setFrameIndex(spilledItr, frameIndex);
  }

  if (mo->isDef()) {

    auto st = getNextMI(mi);
    tii->storeRegToStackSlot(*mi->getParent(), st,
                             usedReg, false, frameIndex, rc, tri);

    auto copyMI = getNextMI(mi);
    for (int i = copyMI->getNumOperands() - 1; i >= 0; --i)
      if (copyMI->getOperand(i).isReg() && copyMI->getOperand(i).getReg() == mo->getReg()) {
        copyMI->getOperand(i).setIsUndef(true);
        break;
      }
    // insert a boundary after store instr.
    /*tii->emitIdemBoundary(*mi->getParent(), getNextMI(copyMI));*/

  } else if (mo->isUse()) {
    assert(frameIndex != INT_MIN);
    tii->loadRegFromStackSlot(*mi->getParent(), mi, usedReg, frameIndex, rc, tri);
    // Inserts a boundary instruction immediately before the load to partition the
    // region into two different parts for avoiding violating idempotence.
    /*auto ld = getPrevMI(mi);*/
    /*insertedMoves.push_back(ld);*/

    /*tii->emitIdemBoundary(*mi->getParent(), ld);*/
  }
}

bool intersects(BitVector lhs, BitVector rhs) {
  for (int idx = lhs.find_first(); idx != -1; idx = lhs.find_next(idx))
    if (!rhs[idx])
      return false;

  return true;
}

typedef std::priority_queue<LiveIntervalIdem *, SmallVector<LiveIntervalIdem *, 64>,
                            llvm::greater_ptr<LiveIntervalIdem>> IntervalMap;

SmallVector<unsigned, 32> regUse;

void IdemRegisterRenamer::processHandledIntervals(std::vector<LiveIntervalIdem *> &handled,
                                                  unsigned currentStart) {
  for (auto interval : handled) {
    if (interval->endNumber() < currentStart) {
      regUse[interval->reg] = 0;
      for (const unsigned *as = tri->getAliasSet(interval->reg); as && *as; ++as)
        regUse[*as] = 0;
    }
  }
}

void IdemRegisterRenamer::assignRegOrStackSlotAtInterval(std::set<unsigned> &allocables,
                                                         LiveIntervalIdem *interval,
                                                         std::vector<LiveIntervalIdem *> &handled,
                                                         std::vector<LiveIntervalIdem *> &spilled) {
  unsigned freeReg = 0;
  for (unsigned reg : allocables) {
    if (!regUse[reg]) {
      freeReg = reg;
      break;
    }
  }
  if (freeReg == 0) {
    // select a handled interval to be spilled out into memory.
    LiveIntervalIdem *spilledItr = 0;
    for (LiveIntervalIdem *itr : handled) {
      if (!spilledItr || itr->costToSpill < spilledItr->costToSpill)
        spilledItr = itr;
    }
    assert(spilledItr && "Must have one interval to be spilled choosen!");
    freeReg = spilledItr->reg;

    getSpilledSubLiveInterval(spilledItr, spilled);
  }

  assert(freeReg != 0 && "No free register found!");
  regUse[freeReg] = 1;
  interval->reg = freeReg;
  insertSpillingCodeForInterval(interval);
}

void IdemRegisterRenamer::revisitSpilledInterval(std::set<unsigned> &allocables,
                                                 std::vector<LiveIntervalIdem *> &spilled) {
  li->releaseMemory();
  li->runOnMachineFunction(*const_cast<MachineFunction *>(mf));

  IntervalMap unhandled;
  std::vector<LiveIntervalIdem *> handled;
  regUse.resize(tri->getNumRegs(), 0);

  for (auto begin = li->interval_begin(), end = li->interval_end(); begin != end; ++begin) {
    handled.push_back(begin->second);
    regUse[begin->second->reg] = 1;
  }

  for (auto begin = spilled.begin(), end = spilled.end(); begin != end; ++begin) {
    unhandled.push(*begin);
  }
  getOrGroupId(spilled);

  while (!unhandled.empty()) {
    auto cur = unhandled.top();
    unhandled.pop();
    /*cur->dump(*const_cast<TargetRegisterInfo *>(tri));*/
    if (!cur->empty()) {
      processHandledIntervals(handled, cur->beginNumber());
    }

    // Allocating another register for current live interval.
    // Note that, only register is allowed to assigned to current interval.
    // Because the current interval corresponds to spilling code.
    std::vector<LiveIntervalIdem *> localSpilled;
    assignRegOrStackSlotAtInterval(allocables, cur, handled, localSpilled);
    getOrGroupId(localSpilled);
    handled.push_back(cur);
  }
}

unsigned IdemRegisterRenamer::tryChooseBlockedRegister(LiveIntervalIdem &interval,
                                                       int useReg,
                                                       BitVector &allocSet) {
  // choose an interval to be evicted into memory, and insert spilling code as
  // appropriate.
  unsigned costMax = INT_MAX;
  LiveIntervalIdem *targetInter = nullptr;
  std::vector<LiveIntervalIdem *> spilledIntervs;

  for (auto physReg = allocSet.find_first(); physReg > 0;
       physReg = allocSet.find_next(physReg)) {
    if (!legalToReplace(physReg, useReg))
      continue;
    assert(li->intervals.count(physReg) && "Why tryChooseFreeRegister does't return it?");
    auto phyItv = li->intervals[physReg];
    IDEM_DEBUG(llvm::errs() << "Found: " << tri->getMinimalPhysRegClass(physReg) << "\n";);

    if (mri->isLiveIn(physReg))
      continue;

    assert(interval.intersects(phyItv) &&
        "should not have interval doesn't interfere with current interval");
    if (phyItv->costToSpill < costMax) {
      costMax = phyItv->costToSpill;
      targetInter = phyItv;
    }
  }

  // no proper interval found to be spilled out.
  if (!targetInter)
    return 0;

  IDEM_DEBUG(llvm::errs() << "Selected evicted physical register is: "
                          << tri->getName(targetInter->reg) << "\n";
                 llvm::errs() << "\nSelected evicted interval is: ";
                 targetInter->dump(*const_cast<TargetRegisterInfo *>(tri)););

  getSpilledSubLiveInterval(targetInter, spilledIntervs);

  if (!spilledIntervs.empty()) {
    std::set<unsigned> allocables;
    getAllocableRegs(targetInter->reg, allocables);
    revisitSpilledInterval(allocables, spilledIntervs);
  }

  for (auto itr : spilledIntervs)
    delete itr;
  return targetInter->reg;
}

/**
 * Backwardly explores all reachable defs of the specified {@code reg} from mi position.
 * @param mi
 * @param reg
 * @param defs
 */
static void findAllReachableDefs(MachineBasicBlock::reverse_iterator begin,
                                 MachineBasicBlock::reverse_iterator end,
                                 MachineBasicBlock *mbb, unsigned reg,
                                 std::vector<MachineInstr*> &defs,
                                 std::set<MachineBasicBlock*> &visited) {
  if (!mbb || !visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    MachineInstr& mi = *begin;
    for (unsigned i = 0, e = mi.getNumOperands(); i < e; i++) {
      auto mo = mi.getOperand(i);
      if (!mo.isReg() || mo.getReg() != reg || !mo.isDef())
        continue;

      defs.push_back(&mi);
      return;
    }
  }

  std::for_each(mbb->pred_begin(), mbb->pred_end(), [&](MachineBasicBlock *pred) {
    findAllReachableDefs(pred->rbegin(), pred->rend(), pred, reg, defs, visited);
  });
}

void IdemRegisterRenamer::spillCurrentUse(AntiDeps &pair) {

  // spill indicates if we have to spill current useMO when
  // tehre is no other free or blocked register available.
  auto useMO = pair.uses.front();

  // Program reach here indicates we can't find any free or blocked register to be used as
  // inserting move instruction.
  std::vector<MachineInstr*> defs;
  std::set<MachineBasicBlock*> visited;
  findAllReachableDefs(MachineBasicBlock::reverse_iterator(useMO.mi),
      useMO.mi->getParent()->rend(),
      useMO.mi->getParent(),
      pair.reg, defs, visited);

  int slotFI = 0;
  const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(pair.reg);
  slotFI = mfi->CreateSpillStackObject(rc->getSize(), rc->getAlignment());
  if (!defs.empty()) {
    for (auto &def : defs) {
      assert(def != def->getParent()->end());
      auto pos = ++MachineBasicBlock::iterator(def);
      tii->storeRegToStackSlot(*def->getParent(), pos, pair.reg, true, slotFI, rc, tri);
    }
  }
  else {
    // The reg must be the live in of entry block of the function.
    assert(mf->front().isLiveIn(pair.reg));
    MachineBasicBlock::iterator pos = mf->front().front();
    tii->storeRegToStackSlot(mf->front(), pos, pair.reg, true, slotFI, rc, tri);
  }

  // Insert a load instruction before the first use.
  auto pos = useMO.mi;
  tii->loadRegFromStackSlot(*pos->getParent(), pos, pair.reg, slotFI, rc, tri);
}

unsigned IdemRegisterRenamer::choosePhysRegForRenaming(MachineOperand *use,
                                                       LiveIntervalIdem *interval,
                                                       DenseSet<unsigned> &unallocableRegs) {
  auto allocSet = tri->getAllocatableSet(*mf);

  // Remove some registers are not available when making decision of choosing.
  for (unsigned i = 0, e = allocSet.size(); i < e; i++)
    if (allocSet[i] && unallocableRegs.count(i))
      allocSet.reset(i);

  // filterUnavailableRegs(use, allocSet, false);

  // obtains a free register used for move instr.
  unsigned useReg = use->getReg();
  unsigned freeReg = tryChooseFreeRegister(*interval, useReg, allocSet);
  if (!freeReg) {
    freeReg = tryChooseBlockedRegister(*interval, useReg, allocSet);
  }

  // If until now, we found no free register, so try to enable flag 'allowsAntiDep'
  // and gives a chance re-handle it.
  /*if (!freeReg) {
    allocSet = tri->getAllocatableSet(*mf);

    // Remove some registers are not available when making decision of choosing.
    filterUnavailableRegs(use, allocSet, true);

    IDEM_DEBUG(llvm::errs() << "Required: " << tri->getMinimalPhysRegClass(use->getReg())->getName() << "\n";);
    // obtains a free register used for move instr.
    freeReg = tryChooseFreeRegister(*interval, useReg, allocSet);
    if (!freeReg) {
      freeReg = tryChooseBlockedRegister(*interval, useReg, allocSet);
    }
  }*/

  if (!freeReg) return 0;
  /*assert(freeReg && "can not to rename the specified register!");*/
  interval->reg = freeReg;
  li->insertOrCreateInterval(freeReg, interval);
  return freeReg;
}

unsigned IdemRegisterRenamer::getFreeRegisterForRenaming(unsigned useReg,
                                                         LiveIntervalIdem *interval,
                                                         DenseSet<unsigned> unallocableRegs) {
  auto allocSet = tri->getAllocatableSet(*mf);

  // Remove some registers are not available when making decision of choosing.
  for (unsigned i = 0, e = allocSet.size(); i < e; i++)
    if (allocSet[i] && unallocableRegs.count(i))
      allocSet.reset(i);

  for (int physReg = allocSet.find_first(); physReg > 0; physReg = allocSet.find_next(physReg)) {
    if (li->intervals.count(physReg)) {

      LiveIntervalIdem *itrv = li->intervals[physReg];

      // we only consider those live interval which doesn't interfere with current
      // interval.
      // No matching in register class should be ignored.
      // Avoiding LiveIn register(such as argument register).
      if (!legalToReplace(physReg, useReg))
        continue;

      if (!itrv->intersects(interval))
        return physReg;
    } else if (legalToReplace(physReg, useReg)) {
      // current physReg is free, so return it.
      return static_cast<unsigned int>(physReg);
    }
  }
  return 0;
}

void IdemRegisterRenamer::walkDFSToGatheringUses(unsigned reg,
                                                 MachineBasicBlock::iterator begin,
                                                 MachineBasicBlock::iterator end,
                                                 MachineBasicBlock *mbb,
                                                 std::set<MachineBasicBlock *> &visited,
                                                 std::vector<MIOp> &usesAndDefs,
                                                 bool &canReplace,
                                                 bool seeIdem) {
  if (!mbb)
    return;
  if (!visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    auto mi = begin;
    bool isTwoAddr = isTwoAddressInstr(mi, reg);
    if (tii->isIdemBoundary(mi)) {
      seeIdem = true;
      continue;
    }

    // when walk through operands, use opr first!
    for (int i = mi->getNumOperands() - 1; i >= 0; i--) {
      auto mo = mi->getOperand(i);
      if (!mo.isReg() || !mo.getReg() || mo.getReg() != reg)
        continue;
      if (mo.isUse()) {
        /*
         * We can't handle such case (the number of predecessor are greater than 1)
         *                 ... = %R0
         *                 /        \
         *               /           \
         *        ... = %R0          BB3
         *     %R0 = ADD %R0, #1      /
         * %R2, %R0 = LDR_INC, %R0   /
         *              \           /
         *               \        /
         *              %R0 = ADD %R0, #1
         *              BX_RET %R0                <------ current mbb
         * we can't replace R0 in current mbb with R3, which will cause
         * wrong semantics when program reach current mbb along with BB3
         */
        if (seeIdem || tii->isReturnInstr(mi) || mbb->pred_size() > 1) {
          canReplace = false;
          return;
        } else
          usesAndDefs.push_back(MIOp(mi, i));
      }
      else {
        // mo.isDef()
        if (isTwoAddr)
          usesAndDefs.push_back(MIOp(mi, i));
        else
          return;
      }
    }
  }

  if (!mbb->succ_empty()) {
    for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ) {
      walkDFSToGatheringUses(reg,
                             (*succ)->begin(),
                             (*succ)->end(),
                             *succ,
                             visited,
                             usesAndDefs,
                             canReplace,
                             seeIdem);
      if (!canReplace)
        return;
    }
  }
}

void IdemRegisterRenamer::getUsesSetOfDef(MachineOperand *def,
                                          std::vector<MIOp> &usesAndDefs,
                                          bool &canReplace) {
  canReplace = true;
  std::set<MachineBasicBlock *> visited;
  auto mbb = def->getParent()->getParent();
  walkDFSToGatheringUses(def->getReg(),
      // Skip current mi defines the def operand, starts walk through from next mi.
      ++MachineBasicBlock::iterator(def->getParent()),
      mbb->end(), mbb, visited, usesAndDefs, canReplace, false);
}

void IdemRegisterRenamer::collectUnallocableRegs(MachineBasicBlock::reverse_iterator begin,
                                                 MachineBasicBlock::reverse_iterator end,
                                                 MachineBasicBlock *mbb,
                                                 std::set<MachineBasicBlock*> &visited,
                                                 DenseSet<unsigned> &unallocableRegs) {
  if (!mbb || !visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    if (tii->isIdemBoundary(&*begin)) {
      auto liveIns = gather->getIdemLiveIns(&*begin);
      /*llvm::errs()<<"Live in: [";
      for (auto r : liveIns)
        llvm::errs()<<tri->getName(r)<<",";
      llvm::errs()<<"]\n";*/

      std::for_each(liveIns.begin(), liveIns.end(), [&](unsigned reg){
        addRegisterWithSubregs(unallocableRegs, reg);
      });
      return;
    }
  }

  std::for_each(mbb->pred_begin(), mbb->pred_end(), [&](MachineBasicBlock *pred){
    collectUnallocableRegs(pred->rbegin(), pred->rend(), pred, visited, unallocableRegs);
  });
}

unsigned IdemRegisterRenamer::getNumFreeRegs(unsigned reg, DenseSet<unsigned> &unallocableRegs) {
  auto allocSet = tri->getAllocatableSet(*mf);

  // Remove some registers are not available when making decision of choosing.
  for (unsigned i = 0, e = allocSet.size(); i < e; i++)
    if (allocSet[i] && unallocableRegs.count(i))
      allocSet.reset(i);
  unsigned res = 0;
  for (int r = allocSet.find_first(); r >= 0; r = allocSet.find_next(r)) {
    if (r != 0 && legalToReplace(r, reg))
      ++res;
  }
  return res;
}

template<typename T>
bool isIntersect(std::vector<T> &lhs, std::vector<T> &rhs) {
  for (auto &elt : lhs) {
    if (std::find(rhs.begin(), rhs.end(), elt) != rhs.end())
      return true;
  }
  return false;
}

bool IdemRegisterRenamer::shouldSpillCurrent(AntiDeps &ad,
                                             DenseSet<unsigned> &unallocableRegs,
                                             std::vector<IdempotentRegion *> &regions) {
  // 1 for the ad which has been removed from antiDeps list.
  unsigned numStimulateousLiveInReg = 1;
  unsigned freeRegs = getNumFreeRegs(ad.reg, unallocableRegs);
  // backup
  auto savedAntiDeps = antiDeps;
  antiDeps.clear();

  std::vector<IdempotentRegion*> rs;
  mir->getRegionsContaining(*ad.uses.front().mi, &rs);
  for (auto r: rs)
    gatherAntiDeps(&r->getEntry());

  numStimulateousLiveInReg = antiDeps.size();
  antiDeps = savedAntiDeps;
  return numStimulateousLiveInReg > freeRegs;
}

void IdemRegisterRenamer::willRenameCauseOtherAntiDep(MachineBasicBlock::iterator begin,
                                                      MachineBasicBlock::iterator end,
                                                      MachineBasicBlock *mbb,
                                                      unsigned reg,
                                                      std::set<MachineBasicBlock *> &visited,
                                                      bool &canRename) {
  if (!mbb || !visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    if (tii->isIdemBoundary(begin))
      return;

    for (int i = begin->getNumOperands() - 1; i >= 0; --i) {
      auto mo = begin->getOperand(i);
      if (!mo.isReg() || mo.getReg() != reg || mo.isUse())
        continue;
      canRename = false;
      return;
    }
  }
  std::for_each(mbb->succ_begin(), mbb->succ_end(), [&](MachineBasicBlock* succ){
    willRenameCauseOtherAntiDep(succ->begin(), succ->end(),
        succ, reg, visited, canRename);
    if (!canRename)
      return;
  });
}

void IdemRegisterRenamer::updateLiveInOfPriorRegions(MachineBasicBlock::reverse_iterator begin,
                                                     MachineBasicBlock::reverse_iterator end,
                                                     MachineBasicBlock *mbb,
                                                     std::set<MachineBasicBlock *> &visited,
                                                     unsigned useReg,
                                                     bool &seenRedef) {
  if (!mbb || !visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {

    if (tii->isIdemBoundary(&*begin)) {
      if (!seenRedef)
        gather->getIdemLiveIns(&*begin).insert(useReg);
    }
    else {
      for (unsigned i = 0, e = begin->getNumOperands(); i < e; i++) {
        auto mo = begin->getOperand(i);
        if (!mo.isReg() || !mo.isDef() || mo.getReg() != useReg)
          continue;

        seenRedef = true;
      }
    }
  }

  std::for_each(mbb->pred_begin(), mbb->pred_end(), [&](MachineBasicBlock *pred) {
    updateLiveInOfPriorRegions(pred->rbegin(), pred->rend(), pred, visited, useReg, seenRedef);
  });
}

bool IdemRegisterRenamer::willRaiseAntiDep(unsigned useReg,
                                           MachineBasicBlock::iterator begin,
                                           MachineBasicBlock::iterator end,
                                           MachineBasicBlock *mbb/*,
                                           std::set<MachineBasicBlock *> &visited*/) {
  if (!mbb)
    return false;

  for (; begin != end; ++begin) {
    if (tii->isIdemBoundary(&*begin))
      return false;
    for (unsigned i = 0, e = begin->getNumOperands(); i < e; i++) {
      auto mo = begin->getOperand(i);
      if (mo.isReg() && mo.isDef() && partialEquals(mo.getReg(), useReg))
        return true;
    }
  }

  for (auto itr = mbb->succ_begin(),succEnd = mbb->succ_end(); itr != succEnd; ++itr){
    auto succ = *itr;
    bool res;
    if (!dt->dominates(succ, mbb))
      res = willRaiseAntiDep(useReg, succ->begin(), succ->end(), succ/*, visited*/);
    else
      res = willRaiseAntiDepsInLoop(useReg, succ->begin(), succ->end(), succ);

    if (res)
      return true;
  }
  return false;
}

bool IdemRegisterRenamer::handleAntiDependences(bool &needRecompute) {
  needRecompute = false;

  if (antiDeps.empty())
    return false;

  std::vector<IdempotentRegion *> regions;

  /*int cnt = 0;*/
  while (!antiDeps.empty()) {
    auto pair = antiDeps.front();
    antiDeps.erase(antiDeps.begin());

    if (pair.uses.empty() || pair.defs.empty())
      continue;

    /*llvm::errs()<<cnt++<<"\n";
    mf->dump();*/

    auto &useMO = pair.uses.front();
    /*if (useMO.mi->getParent()->getName() == "for.end32" && pair.reg == 57) {
      useMO.mi->dump();
    }*/
    mir->getRegionsContaining(*useMO.mi, &regions);

    // get the last insertion position of previous adjacent region
    // or the position of prior instruction depends on if the current instr
    // is a two address instr.
    bool twoAddrInstExits = isTwoAddressInstr(pair.uses.back().mi, pair.reg);

    // Try to replace the old register name with other register to reduce
    // inserted move instruction.
    // If we can not find such register, than alter to insert move.

    // We just count on such situation that all uses are within the same region
    // as the current region.
    std::vector<MIOp> usesAndDef;
    bool canReplace = false;
    // We don't need to cope with those def registers have replaced in the previous round
    size_t i;
    for (i = pair.defs.size(); i > 0; --i) {
      auto &op = pair.defs[i-1];
      if (op.mi->getOperand(op.index).getReg() != pair.reg)
        continue;
      else
        break;
    }
    if (i == 0)
      continue;

    MIOp &miLastDef = pair.defs[i-1];

    getUsesSetOfDef(&miLastDef.mi->getOperand(miLastDef.index), usesAndDef, canReplace);

    auto saved = miLastDef.mi;
    std::set<MachineBasicBlock*> visited;
    if (canReplace)
      willRenameCauseOtherAntiDep(++saved, miLastDef.mi->getParent()->end(),
        miLastDef.mi->getParent(), pair.reg, visited, canReplace);

    // If the current reg is used in ret instr, we can't replace it.
    if (!twoAddrInstExits && canReplace) {
      // We don't replace the name of R0 in ARM and x86 architecture.
      // Because R0 is implicitly used by return instr.

      MachineInstr *mostFarawayMI = nullptr;
      for (auto &mo : usesAndDef) {
        if (!mostFarawayMI || (&*mo.mi != mostFarawayMI &&
            li->getIndex(mo.mi) > li->getIndex(mostFarawayMI)))
          mostFarawayMI = mo.mi;
      }

      // If the last def don't have any uses
      if (usesAndDef.empty()) {
        auto mbb = miLastDef.mi->getParent();
        mostFarawayMI = &mbb->back();
        std::vector<MachineBasicBlock*> worklist;
        visited.clear();
        std::for_each(mbb->succ_begin(), mbb->succ_end(), [&](MachineBasicBlock *succ) {
          worklist.push_back(succ);
        });

        while (!worklist.empty()) {
          mbb = worklist.back();
          worklist.pop_back();
          if (!visited.insert(mbb).second)
            continue;

          if (!mbb->empty()) {
            if (li->getIndex(&mbb->back()) > li->getIndex(mostFarawayMI))
              mostFarawayMI = &mbb->back();
          }
          std::for_each(mbb->succ_begin(), mbb->succ_end(), [&](MachineBasicBlock *succ) {
            worklist.push_back(succ);
          });
        }
      }

      assert(mostFarawayMI);

      auto miDef = pair.defs.front();
      unsigned from = li->getIndex(miDef.mi);
      unsigned to = li->getIndex(mostFarawayMI);

      unsigned phyReg = 0;
      {
        LiveIntervalIdem itrvl;

        // indicates this interval should not be spilled out into memory.
        itrvl.costToSpill = UINT32_MAX;
        std::vector<MIOp> temps;
        temps.assign(pair.defs.begin(), pair.defs.end());

        if (from > to) {
          // this situation could occurs by loop.
          std::swap(from, to);
        }

        itrvl.addRange(from, to);
        for (auto &op : temps) {
          MachineInstr *mi = op.mi;
          /*mi->dump();*/
          itrvl.usePoints.insert(UsePoint(li->getIndex(mi), &mi->getOperand(op.index)));
        }

        for (auto &op : usesAndDef) {
          MachineInstr *mi = op.mi;
          itrvl.usePoints.insert(UsePoint(li->getIndex(mi), &mi->getOperand(op.index)));
        }

        DenseSet<unsigned> unallocableRegs;
        addRegisterWithSubregs(unallocableRegs, pair.reg);
        for (auto &r : regions) {
          auto liveins = gather->getIdemLiveIns(&r->getEntry());
          std::for_each(liveins.begin(), liveins.end(), [&](unsigned reg) {
            addRegisterWithSubregs(unallocableRegs, reg);
          });
        }

        for (MIOp &op : usesAndDef) {
          auto mbb = op.mi->getParent();
          std::for_each(mbb->livein_begin(), mbb->livein_end(), [&](unsigned reg){
            addRegisterWithSubregs(unallocableRegs, reg);
          });
        }

        for (MIOp &op : pair.defs) {
          auto mbb = op.mi->getParent();
          std::for_each(mbb->livein_begin(), mbb->livein_end(), [&](unsigned reg){
            addRegisterWithSubregs(unallocableRegs, reg);
          });

          std::for_each(mbb->pred_begin(), mbb->pred_end(), [&](MachineBasicBlock *pred){
            std::for_each(pred->livein_begin(), pred->livein_end(), [&](unsigned reg){
              addRegisterWithSubregs(unallocableRegs, reg);
            });
          });
        }

        phyReg = getFreeRegisterForRenaming(pair.reg, &itrvl, unallocableRegs);
      }

      if (phyReg != 0) {
        // We have found a free register can be used for replacing the clobber register.
        std::for_each(pair.defs.begin(), pair.defs.end(), [=](MIOp &op) {
          op.mi->getOperand(op.index).setReg(phyReg);
        });
        std::for_each(usesAndDef.begin(), usesAndDef.end(), [=](MIOp &mo) {
          mo.mi->getOperand(mo.index).setReg(phyReg);
        });

        // FIXME There are little problem, add this interval into LiveIntervalAnalysisIdem
        // Finish replacing, skip following inserting move instr.
        li->releaseMemory();
        li->runOnMachineFunction(*mf);

        // Before:
        // 	IDEM
        //  R0 = R0 + R1
        //  R0 = R0 + 1
        //  ret R0
        //
        // After:
        // 	IDEM
        //  R2 = R0 + R1
        //  R0 = R2 + 1
        //  ret R0
        //
        // The anti-dependence on R0 also remains.
        unsigned oldReg = pair.reg;
        for (auto &r : regions) {
          visited.clear();
          auto begin = ++MachineBasicBlock::iterator(r->getEntry());
          collectAntiDepsTrace(oldReg, begin, r->getEntry().getParent()->end(),
                               r->getEntry().getParent(),
                               std::vector<MIOp>(),
                               std::vector<MIOp>(),
                               visited);
        }
        continue;
      }
    }

    // get the free register
    unsigned phyReg = 0;

    /**
     * -------   -------
     * ...=R0    ...=R0
     *   \      /
     *    \    /
     *     v v
     *    ...=R0
     *     ...
     *    R0=...
     *
     * Transformed into:
     *
     * R2 = R0
     * -------   -------
     * ...=R2    ...=R0
     *   \       R2 = R0  (extra move instruction to preserve the program semantics)
     *   \      /
     *    \    /
     *     v v
     *    ...=R2
     *     ...
     *    R0=...
     */
    MIOp *pos = 0;
    for (i = pair.uses.size(); i > 0; --i) {
      MIOp &op = pair.uses[i-1];
      if (op.mi->getOperand(op.index).getReg() == pair.reg) {
        pos = &op;
        break;
      }
    }

    if (pos == nullptr)
      // Reach here, it indicates all uses have been replaced. It isn't needed to advance.
      continue;

    MachineInstr *intervalEnd = pair.uses.back().mi;
    // We should insert a extra move instruction right after the last un-replaced operand
    // pointed by pos.
    if (pos != &pair.uses.back()) {
      auto &back = pair.uses.back();
      unsigned destReg = back.mi->getOperand(back.index).getReg();
      unsigned srcReg = pair.reg;
      MachineBasicBlock::iterator itr = pos->mi->getParent()->end(), insertedPos;
      MachineBasicBlock::iterator begin = pos->mi;

      for (; itr != begin; --itr) {
        insertedPos = itr;
      }
      tii->copyPhysReg(*pos->mi->getParent(), insertedPos, DebugLoc(), destReg, srcReg, true);
      intervalEnd = getPrevMI(insertedPos);
      unsigned index = li->getIndex(itr)-2;
      li->mi2Idx[intervalEnd] = index;
      twoAddrInstExits = false;
    }

    if (!twoAddrInstExits || i > 1) {
      if (regions.empty())
        continue;

      // We shouldn't select the free register from the following kinds:
      // 1. live-in registers of current region.
      // 2. live-in registers of prior region (move instr will be inserted at the end of prior region)
      // 3. interfered registers set.
      //
      // ... = R0 + ...
      // ... = R0 + ...
      // ...
      // R0, ... = LDR_INC R0  (two address instr)
      // we should insert a special move instr for two address instr.
      MachineInstr *insertedPos = nullptr;
      unsigned minIndex = UINT32_MAX;
      insertedPos = nullptr;
      DenseSet<unsigned> unallocableRegs;

      for (auto r : regions) {
        MachineInstr &idem = r->getEntry();
        auto liveins = gather->getIdemLiveIns(&idem);
        std::for_each(liveins.begin(), liveins.end(), [&](unsigned reg) {
          addRegisterWithSubregs(unallocableRegs, reg);
        });

        auto begin = MachineBasicBlock::reverse_iterator(idem);
        visited.clear();
        collectUnallocableRegs(begin, idem.getParent()->rend(),
            idem.getParent(), visited, unallocableRegs);

        unsigned index = li->getIndex(&idem);
        if (index < minIndex) {
          minIndex = index;
          insertedPos = &idem;
        }
      }

      // can not assign the old register to use mi
      addRegisterWithSubregs(unallocableRegs, pair.reg);
      auto begin = pair.uses.begin(), end = pair.uses.begin() + i;
      std::for_each(begin, end, [&](MIOp &op) {
        MachineInstr *useMI = op.mi;
        for (unsigned j = 0, e = useMI->getNumOperands(); j < e; j++) {
          auto mo = useMI->getOperand(j);
          if (mo.isReg() && mo.getReg() && mo.isDef())
            addRegisterWithSubregs(unallocableRegs, mo.getReg());
        }
      });

      /**
       * Avoiding repeatedly erase and add anti-dependence like following example.
       * ... = R1
       *   ...
       * R1 = ...
       * R3 = ...
       *
       * ==>
       *
       * ... = R3
       *   ...
       * R1 = ...
       * R3 = ...
       *
       * It will repeat the process, we should avoid that situation.
       */
      auto miOp = pair.uses.front();
      LiveIntervalIdem interval;

      // indicates this interval should not be spilled out into memory.
      interval.costToSpill = UINT32_MAX;

      auto from = li->getIndex(insertedPos) - 2;
      auto to = li->getIndex(intervalEnd);

      if (from > to) {
        // this situation could occurs by loop.
        std::swap(from, to);
      }

      interval.addRange(from, to);    // add an interval for a temporal move instr.

      do {
        if (shouldSpillCurrent(pair, unallocableRegs, regions)) {
          spillCurrentUse(pair);
          goto UPDATE_INTERVAL;
        }

        phyReg = choosePhysRegForRenaming(&miOp.mi->getOperand(miOp.index), &interval, unallocableRegs);

        if (!phyReg)
          break;

        if (!willRaiseAntiDep(phyReg, useMO.mi, useMO.mi->getParent()->end(), useMO.mi->getParent()))
          break;

        addRegisterWithSubregs(unallocableRegs, phyReg);
      }while (true);

      if (!phyReg) {
        spillCurrentUse(pair);
        goto UPDATE_INTERVAL;
      }

      assert(insertedPos);

      // FIXME 10/23/2018
      // li->intervals.insert(std::make_pair(phyReg, interval));

      assert(TargetRegisterInfo::isPhysicalRegister(phyReg));
      assert(phyReg != pair.reg);

      size_t e = pair.uses.size() - 1;
      size_t limit = twoAddrInstExits ? e : i;

      for (size_t k = 0; k < limit; k++)
        pair.uses[k].mi->getOperand(pair.uses[k].index).setReg(phyReg);

      tii->copyPhysReg(*insertedPos->getParent(), insertedPos, DebugLoc(), phyReg, pair.reg, true);

      // Note that, after replace the old register with new reg, which maybe
      // raise other anti-dependence, such as:
      // R0 = R2 + R1
      // R2 = ....
      // R3 = ....
      // replace R2 in first instr with R3 will cause another anti-dependence
      // R0 = R3 + R1
      // R2 = ...
      // R3 = ...
      for (auto &r : regions) {
        auto &liveIns = gather->getIdemLiveIns(&r->getEntry());
        liveIns.erase(pair.reg);
        liveIns.insert(phyReg);

        /*llvm::errs()<<"[";
        for (auto r : liveIns)
          llvm::errs()<<tri->getName(r)<<",";
        llvm::errs()<<"]\n";

        mf->dump();*/

        // for an iteration of each live-in register, renew the visited set.
        auto beginItr = ++MachineBasicBlock::iterator(r->getEntry());
        visited.clear();
        collectAntiDepsTrace(phyReg, beginItr, r->getEntry().getParent()->end(),
                             r->getEntry().getParent(),
                             std::vector<MIOp>(),
                             std::vector<MIOp>(), visited);

        // FIXME Update the live in registers for the region where insertedPos instr resides
        auto mbb = r->getEntry().getParent();
        visited.clear();
        bool seenRedef = false;
        updateLiveInOfPriorRegions(MachineBasicBlock::reverse_iterator(r->getEntry()),
                                   mbb->rend(), mbb, visited, pair.reg, seenRedef);
      }
    }

    // Now, replace all old registers used in two addr instr with the new register.
    if (twoAddrInstExits) {
      auto useMI = pair.uses.back().mi;
      auto moIndex = pair.uses.back().index;
      auto mbb = useMI->getParent();
      unsigned oldReg = pair.reg;

      // request a new register
      if (phyReg == 0) {

        // We have to check if it will cause anti-dependence after replacing
        // the defined reg of two addr instr, for example:
        //
        // IDEM
        // R0 = STr R1, R0
        // ... = R0
        // ...
        // R0 = ...
        //
        // After replacing,
        // IDEM
        // R2 = R0
        // R2 = STr R1, R2
        // ... = R0
        // ...
        // R0 = ...
        auto savedAntiDeps = antiDeps;
        antiDeps.clear();
        std::vector<MIOp> uses, defs;
        visited.clear();
        collectAntiDepsTrace(pair.reg, ++MachineBasicBlock::iterator(useMI),
                             mbb->end(), mbb, uses, defs, visited);

        bool causeAntiDep = !antiDeps.empty();
        antiDeps = savedAntiDeps;
        if (causeAntiDep) {
          spillCurrentUse(pair);
          goto UPDATE_INTERVAL;
        }

        DenseSet<unsigned> unallocableRegs;
        for (unsigned j = 0, e = useMI->getNumOperands(); j < e; j++) {
          auto mo = useMI->getOperand(j);
          if (mo.isReg() && mo.getReg())
            addRegisterWithSubregs(unallocableRegs, mo.getReg());
        }
        unallocableRegs.insert(oldReg);

        for (auto & r : regions) {
          auto liveins = gather->getIdemLiveIns(&r->getEntry());
          std::for_each(liveins.begin(), liveins.end(), [&](unsigned reg) {
            addRegisterWithSubregs(unallocableRegs, reg);
          });
        }

        LiveIntervalIdem interval;

        // indicates this interval should not be spilled out into memory.
        interval.costToSpill = UINT32_MAX;

        auto from = li->getIndex(useMI) - 2;
        auto to = li->getIndex(pair.uses.back().mi);

        interval.addRange(from, to);    // add an interval for a temporal move instr.
        auto miOp = pair.uses.front();
        phyReg = choosePhysRegForRenaming(&miOp.mi->getOperand(miOp.index), &interval, unallocableRegs);

        if (!phyReg) {
          spillCurrentUse(pair);
          goto UPDATE_INTERVAL;
        }

        // FIXME 10/23/2018
        // li->intervals.insert(std::make_pair(phyReg, interval));
        assert(TargetRegisterInfo::isPhysicalRegister(phyReg));
        assert(phyReg != oldReg);

        // We must insert move firstly, and than substitute the old reg with new reg.
        tii->copyPhysReg(*mbb, useMI, DebugLoc(), phyReg,  oldReg, useMI->getOperand(moIndex).isKill());

        for (unsigned i = 0, e = useMI->getNumOperands(); i < e; i++) {
          MachineOperand& mo = useMI->getOperand(i);
          if (mo.isReg() && mo.getReg() == oldReg)
            mo.setReg(phyReg);
        }
      }
      else {

        // Step#8: substitute the old reg with phyReg,
        // and remove other anti-dep on this use.

        // We must insert move firstly, and than substitute the old reg with new reg.
        tii->copyPhysReg(*mbb, useMI, DebugLoc(), oldReg, phyReg, true);
      }
    }

    {
      std::vector<std::vector<AntiDeps>::iterator> toRemoved;
      for (auto itr = antiDeps.begin(), end = antiDeps.end(); itr != end; ++itr) {
        if (itr->reg == pair.reg) {
          bool found = false;
          for (auto &op : itr->uses) {
            if (std::find(pair.uses.begin(), pair.uses.end(), op) != pair.uses.end()) {
              toRemoved.push_back(itr);
              needRecompute = true;
              found = true;
              break;
            }
          }
          if (!found) {
            for (auto &op : itr->defs)
              if (std::find(pair.defs.begin(), pair.defs.end(), op) != pair.defs.end()) {
                toRemoved.push_back(itr);
                needRecompute = true;
                break;
              }
          }
        }
      }
      for (auto &itr : toRemoved)
        antiDeps.erase(itr);
    }

UPDATE_INTERVAL:
  // FIXME, use an lightweight method to update LiveIntervalAnalysisIdem
  li->releaseMemory();
  li->runOnMachineFunction(*mf);
  /*delete gather;
  gather = new LiveInsGather(*mf);
  gather->run();*/
  }
  return true;
}

bool IdemRegisterRenamer::runOnMachineFunction(MachineFunction &MF) {
  // Step#1: Compute the live-in registers set for each idempotence region.
  // Step#2: Handle two address instruction, insert a move instr before it right away.
  // Step#3: Determine whether I should insert a move instr for anti-dependence or
  //         replace the register name of anti-dependence.
  //
  //         When there are enough free registers, we take the method that
  //         replacing the register name rather than inserting a move instr in
  //         in the previous region(when no such region, create a new region before current region)
  gather = new LiveInsGather(MF);
  mir = getAnalysisIfAvailable<MachineIdempotentRegions>();
  assert(mir && "No MachineIdempotentRegions available!");
  li = getAnalysisIfAvailable<LiveIntervalAnalysisIdem>();
  assert(li);

  dt = getAnalysisIfAvailable<MachineDominatorTree>();

  tii = MF.getTarget().getInstrInfo();
  tri = MF.getTarget().getRegisterInfo();
  mf = &MF;
  mri = &MF.getRegInfo();
  mfi = MF.getFrameInfo();

  // Collects anti-dependences operand pair.
  /*llvm::errs() << "Before renaming2: \n";
  MF.dump();*/

  bool changed = false;
  bool needRecompute;

  do {
    collectLiveInRegistersForRegions();
    computeAntiDependenceSet();
    changed |= handleAntiDependences(needRecompute);
    if (!needRecompute)
      break;
  }while (true);

  /*llvm::errs() << "After renaming2: \n";
  MF.dump();*/
  clear();
  return changed;
}
