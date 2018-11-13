#ifndef LLVM_IDEMREGISTERRENAMING2_H
#define LLVM_IDEMREGISTERRENAMING2_H

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
#include <llvm/Support/Timer.h>
#include "llvm/Support/Debug.h"

#include "IdemUtil.h"
#include "IdempotentRegionLiveInsGather.h"
#include "LiveIntervalAnalysisIdem.h"
#include "MoveResolver.h"
#include "VirRegRewriter.h"

#include <queue>
#include <utility>

/*static cl::opt<bool> EnableIdemTimeStatistic("idem-time-statistic",
                                             cl::init(false),
                                             cl::desc("Enable time statistic in idem renaming"),
                                             cl::Hidden);*/

/// @author Jianping Zeng.
namespace llvm {

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

typedef std::priority_queue<LiveIntervalIdem *, SmallVector<LiveIntervalIdem *, 64>,
                            llvm::greater_ptr<LiveIntervalIdem>> IntervalMap;

class MoveResolver;

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
    ml = nullptr;
    cur = nullptr;
    rewriter = nullptr;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalAnalysisIdem>();
    AU.addRequired<MachineIdempotentRegions>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
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
    dt = nullptr;
    ml = nullptr;
    antiDeps.clear();
    region2NumAntiDeps.clear();
    interval2StackSlotMap.clear();
    if (rewriter)
      delete rewriter;
  }

public:
  inline void collectLiveInRegistersForRegions();
  void computeAntiDependenceSet();
  void gatherAntiDeps(MachineInstr *idem);
  bool handleAntiDependences();
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

  void spillCurrentUse(AntiDeps &pair,
                       MachineInstr *insertedPos,
                       DenseSet<unsigned> &unallocableRegs);

  void choosePhysRegForRenaming(unsigned useReg,
                                LiveIntervalIdem *interval,
                                DenseSet<unsigned> &unallocableRegs);

  void resolveDataflow();

  void findPosAndInsertMove(MachineBasicBlock *src, MachineBasicBlock *dst);

  bool legalToReplace(unsigned newReg, unsigned oldReg);
  unsigned tryChooseFreeRegister(LiveIntervalIdem &interval,
                                 int useReg,
                                 BitVector &allocSet);
  void initializeIntervalSet();

  void prehandled(unsigned position);

  unsigned allocateBlockedRegister(LiveIntervalIdem *interval,
                                   BitVector &allocSet);

  /**
   * @code
   * split an interval at the optimal position between minSplitPos and
   * maxSplitPos in two parts:
   * 1) the left part has already a location assigned
   * 2) the right part is always on the stack and therefore ignored in further processing
   * @endcode
   *
   * @param it
   * @param startPos
   */
  void splitForSpilling(LiveIntervalIdem *it, unsigned startPos);

  void splitAndSpill(LiveIntervalIdem *it, unsigned startPos, unsigned endPos, bool isActive);

  LiveIntervalIdem *splitBeforeUsage(LiveIntervalIdem *it, unsigned minSplitPos, unsigned maxSplitPos);

  unsigned findOptimalSplitPos(LiveIntervalIdem *it, unsigned minSplitPos, unsigned maxSplitPos);

  unsigned findOptimalSplitPos(MachineBasicBlock *minBlock, MachineBasicBlock *maxBlock,
                               unsigned maxSplitPos);

  void linearScan(BitVector &allocSet);

  unsigned getFreePhyReg(LiveIntervalIdem *interval, BitVector &allocSet);

  int assignInterval2StackSlot(LiveIntervalIdem *interval);

  void insertMove(unsigned insertedPos, LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt);

  LiveIntervalIdem *splitIntervalWhenPartialAvailable(LiveIntervalIdem *it, unsigned regAvaiUntil);

  bool getSpilledSubLiveInterval(LiveIntervalIdem *interval,
                                 std::vector<LiveIntervalIdem *> &spilledItrs);
  void getAllocableRegs(unsigned useReg, std::set<unsigned> &allocables);

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
                              std::vector<MIOp> uses,
                              std::vector<MIOp> defs,
                              bool &canReplace,
                              bool seeIdem);

  void collectUnallocableRegs(AntiDeps &pair,
                              MachineInstr *&insertedPos,
                              std::vector<IdempotentRegion *> &regions,
                              DenseSet<unsigned> &unallocableRegs);

  void collectUnallocableRegsDFS(MachineBasicBlock::reverse_iterator begin,
                                 MachineBasicBlock::reverse_iterator end,
                                 MachineBasicBlock *mbb,
                                 std::set<MachineBasicBlock *> &visited,
                                 DenseSet<unsigned> &unallocableRegs);

  unsigned getNumFreeRegs(unsigned reg, DenseSet<unsigned> &unallocableRegs);

  bool shouldSpillCurrent(unsigned reg,
                          DenseSet<unsigned> &unallocableRegs,
                          std::vector<IdempotentRegion *> &regions);

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

  void countRegistersRaiseAntiDep(MachineBasicBlock::iterator begin,
                                  MachineBasicBlock::iterator end,
                                  MachineBasicBlock *mbb,
                                  DenseSet<unsigned> &unallocableRegs);

  /**
   * This function differs from {@code countRegisterRaiseAntiDep} in the aspect of
   * this function is about to check whether the introduced defined register will
   * raise a new anti-dependence.
   * @param mi
   * @param unallocableRegs
   */
  void countDefRegRaiseAntiDep(MachineInstr *mi, DenseSet<unsigned> &unallocableRegs) {
    if (!mi) return;

    typedef MachineBasicBlock::reverse_iterator Itr;
    struct WorkItem {
      Itr begin, end;
      MachineBasicBlock *mbb;
      WorkItem(Itr _begin, Itr _end, MachineBasicBlock *_mbb) :
          begin(_begin), end(_end), mbb(_mbb) {}
    };

    std::vector<WorkItem> worklist;
    std::set<MachineBasicBlock*> visited;
    MachineBasicBlock *mbb = mi->getParent();
    worklist.emplace_back(Itr(mi), mbb->rend(), mbb);

    while (!worklist.empty()) {
      auto cur = worklist.back();
      worklist.pop_back();
      mbb = cur.mbb;
      visited.insert(mbb);

      auto begin = cur.begin;
      auto end = cur.end;
      for (; begin != end; ++begin) {
        if (tii->isIdemBoundary(&*begin))
          break;
      }
      // idem exists.
      if (begin != end) {
        auto &buf = gather->getIdemLiveIns(&*begin);
        std::for_each(buf.begin(), buf.end(), [&](unsigned r) {
          addRegisterWithSubregs(unallocableRegs, r);
          addRegisterWithSuperRegs(unallocableRegs, r);
        });
      }
      else {
        std::vector<MachineBasicBlock*> buf;
        buf.assign(mbb->pred_begin(), mbb->pred_end());
        std::for_each(buf.rbegin(), buf.rend(), [&](MachineBasicBlock *pred) {
          if (!visited.count(pred))
            worklist.emplace_back(pred->rbegin(), pred->rend(), pred);
        });
      }
    }
  }

  void countRegisterRaiseAntiDepsInLoop(const MachineBasicBlock::iterator &idem,
                                        const MachineBasicBlock::iterator &end,
                                        MachineBasicBlock *mbb,
                                        DenseSet<unsigned> &unallocableRegs);

  void computeAntiDepsInLoop(unsigned reg,
                             const MachineBasicBlock::iterator &idem,
                             const MachineBasicBlock::iterator &end,
                             MachineBasicBlock *mbb,
                             std::vector<MIOp> uses,
                             std::vector<MIOp> defs);

  void addRegisterWithSubregs(DenseSet<unsigned> &set, unsigned reg) {
    set.insert(reg);
    if (!TargetRegisterInfo::isStackSlot(reg) &&
        TargetRegisterInfo::isPhysicalRegister(reg)) {
      for (const unsigned *r = tri->getSubRegisters(reg); *r; ++r)
        set.insert(*r);
    }
  }

  void addRegisterWithSuperRegs(DenseSet<unsigned> &set, unsigned reg) {
    set.insert(reg);
    if (!TargetRegisterInfo::isStackSlot(reg) &&
        TargetRegisterInfo::isPhysicalRegister(reg)) {
      for (const unsigned *r = tri->getSuperRegisters(reg); *r; ++r)
        set.insert(*r);
    }
  }

  bool isAssignedPhyReg(LiveIntervalIdem *it) {
    return interval2AssignedRegMap.count(it);
  }

  bool isAssignedStackSlot(LiveIntervalIdem *it) {
    return it && interval2StackSlotMap.count(it);
  }

  unsigned getAssignedPhyReg(LiveIntervalIdem *it) {
    assert(isAssignedPhyReg(it));
    return interval2AssignedRegMap[it];
  }

  int getAssignedStackSlot(LiveIntervalIdem *it) {
    assert(isAssignedStackSlot(it));
    return interval2StackSlotMap[it];
  }

  void emitRegToReg(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MI, DebugLoc DL,
                    unsigned DestReg, unsigned SrcReg,
                    bool KillSrc);

  void eliminatePseudoMoves();

private:
  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;
  MachineIdempotentRegions *mir;
  std::deque<AntiDeps> antiDeps;
  LiveInsGather *gather;
  LiveIntervalAnalysisIdem *li;
  MachineFunction *mf;
  MachineRegisterInfo *mri;
  MachineFrameInfo *mfi;
  MachineDominatorTree *dt;
  MachineLoopInfo *ml;

  /**
   * This map used for recording the number of anti-dependencies for each
   * idmepotent region indicated by idem instruction.
   */
  std::map<MachineInstr*, size_t> region2NumAntiDeps;
  BitVector reservedRegs;
  IntervalMap unhandled;
  std::vector<LiveIntervalIdem *> handled;
  std::vector<LiveIntervalIdem *> active;
  std::vector<LiveIntervalIdem *> inactive;
  LiveIntervalIdem *cur;
  std::map<LiveIntervalIdem*, unsigned> interval2AssignedRegMap;
  std::map<LiveIntervalIdem*, int> interval2StackSlotMap;

  llvm::MoveResolver *resolver;
  VirRegRewriter *rewriter;
  // Those move instruction should be rewrited into real copy after renaming
  std::vector<MachineInstr*> pseduoMoves;
};
}

#endif //LLVM_IDEMREGISTERRENAMING2_H
