#include <utility>

//===----- IdemRegisterRenaming.cpp - Register regnaming after RA ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-renaming"

#include <llvm/PassSupport.h>
#include <llvm/CodeGen/MachineIdempotentRegions.h>
#include <queue>
#include <llvm/ADT/SetOperations.h>
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "LiveIntervalAnalysisIdem.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "IdemUtil.h"
#include "IdempotentRegionLiveInsGather.h"

using namespace llvm;

/// @author Jianping Zeng.
namespace {

struct AntiDeps {
  unsigned reg;
  std::vector<MachineOperand *> uses;
  std::vector<MachineOperand *> defs;
};

class IdemRegisterRenamer : public MachineFunctionPass {
public:
  static char ID;
  IdemRegisterRenamer() : MachineFunctionPass(ID) {
    initializeIdemRegisterRenamerPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalAnalysisIdem>();
    AU.addRequired<MachineIdempotentRegions>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const {
    return "Register Renaming for Idempotence pass";
  }

  virtual void releaseMemory() {
    tii = nullptr;
    tri = nullptr;
    delete mir;
    reversePostOrderMBBs.clear();
    antiDeps.clear();
    delete gather;
    delete li;
    mf = nullptr;
    mri = nullptr;
    mfi = nullptr;
  }

private:
  inline void collectLiveInRegistersForRegions();
  void computeAntiDependenceSet();
  void gatherAntiDeps(MachineInstr *idem);
  bool handleAntiDependences();
  void collectAntiDepsTrace(unsigned,
                            MachineBasicBlock::iterator,
                            MachineBasicBlock::iterator,
                            MachineBasicBlock *,
                            std::set<MachineBasicBlock *> &,
                            std::vector<MachineOperand *>,
                            std::vector<MachineOperand *>);
  bool isTwoAddressInstr(MachineInstr *useMI);
  unsigned choosePhysRegForRenaming(MachineOperand *use,
                                    LiveIntervalIdem *interval,
                                    DenseSet<unsigned> &unallocableRegs);
  void filterUnavailableRegs(MachineOperand *use,
                             BitVector &allocSet,
                             bool allowsAntiDep);
  bool legalToReplace(unsigned physReg, int reg);
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
                       std::vector<MachineOperand *> &uses,
                       bool &usesInSameRegion);

  unsigned getFreeRegisterForRenaming(unsigned useReg,
                                      LiveIntervalIdem *interval,
                                      DenseSet<unsigned> unallocableRegs);

  void walkDFSToGatheringUses(unsigned reg,
                              MachineBasicBlock::iterator begin,
                              MachineBasicBlock::iterator end,
                              MachineBasicBlock *mbb,
                              std::set<MachineBasicBlock*> &visited,
                              std::vector<MachineOperand *> &uses,
                              bool &usesInSameRegion,
                              bool seeIdem);
private:
  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;
  MachineIdempotentRegions *mir;
  std::vector<MachineBasicBlock *> reversePostOrderMBBs;
  std::vector<AntiDeps> antiDeps;
  LiveInsGather *gather;
  LiveIntervalAnalysisIdem *li;
  MachineFunction *mf;
  MachineRegisterInfo *mri;
  MachineFrameInfo *mfi;
};
}

INITIALIZE_PASS_BEGIN(IdemRegisterRenamer, "reg-renaming",
                      "Register Renaming for Idempotence", false, false)
  INITIALIZE_PASS_DEPENDENCY(LiveIntervalAnalysisIdem)
  INITIALIZE_PASS_DEPENDENCY(MachineIdempotentRegions)
INITIALIZE_PASS_END(IdemRegisterRenamer, "reg-renaming",
                    "Register Renaming for Idempotence", false, false)

char IdemRegisterRenamer::ID = 0;

void IdemRegisterRenamer::collectLiveInRegistersForRegions() {
  gather->run();
}

static bool contains(std::vector<MachineOperand *> &set, int reg) {

}

void IdemRegisterRenamer::collectAntiDepsTrace(unsigned reg,
                                               MachineBasicBlock::iterator idem,
                                               MachineBasicBlock::iterator end,
                                               MachineBasicBlock *mbb,
                                               std::set<MachineBasicBlock *> &visited,
                                               std::vector<MachineOperand *> uses,
                                               std::vector<MachineOperand *> defs) {
  if (!visited.insert(mbb).second)
    return;

  for (auto itr = idem; itr != end; ++itr) {
    if (tii->isIdemBoundary(itr))
      return;

    for (int i = 0, e = itr->getNumOperands(); i < e; i++) {
      auto mo = itr->getOperand(i);
      if (!mo.isReg() || !mo.getReg())
        continue;

      if (mo.isUse()) {
        if (contains(defs, mo.getReg()))
          goto CONST_IDEM;

        uses.push_back(&mo);
      } else
        defs.push_back(&mo);
    }
  }

  if (mbb && !mbb->succ_empty()) {
    for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ) {
      collectAntiDepsTrace(reg, (*succ)->begin(), (*succ)->end(), *succ, visited, uses, defs);
    }
  }

  // Construct anti-dependences accoridng uses and defs set.
  CONST_IDEM:
  antiDeps.push_back({reg, uses, defs});
  return;
}

void IdemRegisterRenamer::gatherAntiDeps(MachineInstr *idem) {
  auto liveIns = gather->getIdemLiveIns(idem);
  if (liveIns.empty())
    return;

  std::set<MachineBasicBlock *> visited;
  for (auto reg : liveIns) {
    collectAntiDepsTrace(reg, MachineBasicBlock::iterator(idem),
                         idem->getParent()->end(), idem->getParent(),
                         visited, std::vector<MachineOperand *>(),
                         std::vector<MachineOperand *>());
  }
}

void IdemRegisterRenamer::computeAntiDependenceSet() {
  for (auto &itr : *mir) {
    MachineInstr *idem = &itr->getEntry();
    assert(idem && tii->isIdemBoundary(idem));
    gatherAntiDeps(idem);
  }
}

bool IdemRegisterRenamer::isTwoAddressInstr(MachineInstr *useMI) {
  // We should not rename the two-address instruction.
  auto MCID = useMI->getDesc();
  int numOps = useMI->isInlineAsm() ? useMI->getNumOperands() : MCID.getNumOperands();
  for (int i = 0; i < numOps; i++) {
    unsigned destIdx;
    if (!useMI->isRegTiedToDefOperand(i, &destIdx))
      continue;

    return true;
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

bool IdemRegisterRenamer::legalToReplace(unsigned physReg, int reg) {
  for (unsigned i = 0, e = tri->getNumRegClasses(); i < e; i++) {
    auto rc = tri->getRegClass(i);
    if (rc->contains(physReg) && rc->contains(reg))
      return true;
  }
  return false;
}

void IdemRegisterRenamer::filterUnavailableRegs(MachineOperand *use,
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

static MachineInstr *getPrevMI(MachineInstr *mi) {
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
    cur->dump(*const_cast<TargetRegisterInfo *>(tri));
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

  assert(freeReg && "can not to rename the specified register!");
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
                            std::set<MachineBasicBlock*> &visited,
                            std::vector<MachineOperand *> &uses,
                            bool &usesInSameRegion,
                            bool seeIdem) {
  if (!mbb) return;
  if (!visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    auto mi = begin;
    if (tii->isIdemBoundary(mi)) {
      seeIdem = true;
      continue;
    }

    for (int i = 0, e = mi->getNumOperands(); i < e; i++) {
      auto mo = mi->getOperand(i);
      if (mo.isReg() && mo.isUse() && mo.getReg() == reg) {
        if (seeIdem) {
          usesInSameRegion = false;
          return;
        }
        else
          uses.push_back(&mo);
      }
    }
  }

  if (!mbb->succ_empty()) {
    for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ) {
      walkDFSToGatheringUses(reg, (*succ)->begin(), (*succ)->end(), *succ, visited, uses, usesInSameRegion, seeIdem);
      if (!usesInSameRegion) return;
    }
  }
}

void IdemRegisterRenamer::getUsesSetOfDef(MachineOperand *def,
                                          std::vector<MachineOperand *> &uses,
                                          bool &usesInSameRegion) {
  usesInSameRegion = true;
  std::set<MachineBasicBlock*> visited;
  auto mbb = def->getParent()->getParent();
  walkDFSToGatheringUses(def->getReg(), def->getParent(), mbb->end(), mbb, visited, uses, usesInSameRegion, false);
}

bool IdemRegisterRenamer::handleAntiDependences() {
  if (antiDeps.empty())
    return false;

  std::vector<IdempotentRegion *> regions;

  for (auto &pair : antiDeps) {

    mir->getRegionsContaining(*pair.uses.front()->getParent(), &regions);

    // get the last insertion position of previous adjacent region
    // or the position of prior instruction depends on if the current instr
    // is a two address instr.
    bool twoAddrInstExits = isTwoAddressInstr(pair.uses.back()->getParent());
    // Try to replace the old register name with other register to reduce
    // inserted move instruction.
    // If we can not find such register, than alter to insert move.
    if (!twoAddrInstExits) {
      // We just count on such situation that all uses are within the same region
      // as the current region.
      std::vector<MachineOperand *> uses;
      bool usesInSameRegion = false;
      getUsesSetOfDef(pair.defs.back(), uses, usesInSameRegion);
      if (!usesInSameRegion)
        goto INSERT_MOVE;

      MachineOperand *mostFarway = nullptr;
      for (auto &mo : uses) {
        if (!mostFarway || li->getIndex(mo->getParent()) > li->getIndex(mostFarway->getParent()))
          mostFarway = mo;
      }

      unsigned from = li->getIndex(pair.defs[0]->getParent());
      unsigned to = li->getIndex(mostFarway->getParent());
      LiveIntervalIdem *itrvl = new LiveIntervalIdem();
      itrvl->addRange(from, to);
      std::for_each(pair.defs.begin(), pair.defs.end(), [&](MachineOperand *mo) {
        itrvl->usePoints.insert(UsePoint(li->getIndex(mo->getParent()), mo));
      });

      std::for_each(uses.begin(), uses.end(), [&](MachineOperand *mo) {
        itrvl->usePoints.insert(UsePoint(li->getIndex(mo->getParent()), mo));
      });


      DenseSet<unsigned> unallocableRegs;
      unallocableRegs.insert(pair.reg);
      for (auto &r : regions) {
        set_union(unallocableRegs, gather->getIdemLiveIns(&r->getEntry()));
      }

      unsigned phyReg = getFreeRegisterForRenaming(pair.reg, itrvl, unallocableRegs);

      if (phyReg != 0) {
        // Find a free register can be used for replacing the clobber register.
        std::for_each(pair.defs.begin(), pair.defs.end(), [=](MachineOperand *mo) {
          mo->setReg(phyReg);
        });
        std::for_each(uses.begin(), uses.end(), [=](MachineOperand *mo) {
          mo->setReg(phyReg);
        });

        // Finish replacing, skip following inserting move instr.
        continue;
      }
    }

INSERT_MOVE:
    // get the free register
    unsigned phyReg = 0;

    if (!twoAddrInstExits || pair.uses.size() > 1) {
      if (regions.empty())
        continue;

      // We choose that insertion whose index is minimal
      unsigned minIndex = UINT32_MAX;
      MachineInstr *insertedPos = nullptr;
      std::vector<MachineInstr *> prevRegionIdems;

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
      DenseSet<unsigned> unallocableRegs;

      for (auto r : regions) {
        MachineInstr &idem = r->getEntry();
        set_union(unallocableRegs, gather->getIdemLiveIns(&idem));

        unsigned index = li->getIndex(&idem);
        if (index < minIndex) {
          minIndex = index;
          insertedPos = &idem;
        }
      }

      assert(insertedPos);
      mir->getRegionsContaining(*getPrevMI(insertedPos), &regions);
      for (auto r : regions)
        set_union(unallocableRegs, gather->getIdemLiveIns(&r->getEntry()));

      // can not assign the old register to use mi
      unallocableRegs.insert(pair.reg);
      std::for_each(pair.uses.begin(), pair.uses.end(), [&](MachineOperand *mo) {
        MachineInstr *useMI = mo->getParent();
        for (unsigned i = 0, e = useMI->getNumOperands(); i < e; i++)
          if (useMI->getOperand(i).isReg() && useMI->getOperand(i).getReg() &&
              useMI->getOperand(i).isDef())
            unallocableRegs.insert(useMI->getOperand(i).getReg());
      });

      LiveIntervalIdem *interval = new LiveIntervalIdem;

      // indicates this interval should not be spilled out into memory.
      interval->costToSpill = UINT32_MAX;

      auto from = li->getIndex(insertedPos) - 2;
      auto to = li->getIndex(pair.uses.back()->getParent());

      interval->addRange(from, to);    // add an interval for a temporal move instr.
      phyReg = choosePhysRegForRenaming(pair.uses[0], interval, unallocableRegs);

      li->intervals.insert(std::make_pair(phyReg, interval));

      assert(TargetRegisterInfo::isPhysicalRegister(phyReg));
      assert(phyReg != pair.uses.back()->getReg());
    }

    // Now, replace all old registers used in useMI with the new register.
    for (size_t i = 0, e = pair.uses.size() - 1; i < e; i++)
      pair.uses[i]->setReg(phyReg);

    if (twoAddrInstExits) {
      auto useMI = pair.uses.back()->getParent();
      auto mbb = useMI->getParent();

      // Step#8: substitute the old reg with phyReg,
      // and remove other anti-dep on this use.
      unsigned oldReg = pair.reg;
      for (int i = 0, e = useMI->getNumOperands(); i < e; i++) {
        auto mo = useMI->getOperand(i);
        if (mo.isReg() && mo.getReg() == oldReg)
          mo.setReg(phyReg);
      }
      tii->copyPhysReg(*mbb, useMI, DebugLoc(), phyReg, oldReg, pair.uses.back()->isKill());
    }
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

  tii = MF.getTarget().getInstrInfo();
  tri = MF.getTarget().getRegisterInfo();
  mf = &MF;
  mri = &MF.getRegInfo();
  mfi = MF.getFrameInfo();

  // Collects anti-dependences operand pair.
  collectLiveInRegistersForRegions();
  computeReversePostOrder(MF, reversePostOrderMBBs);

  computeAntiDependenceSet();

  bool changed = false;
  changed |= handleAntiDependences();
  delete gather;
  return changed;
}