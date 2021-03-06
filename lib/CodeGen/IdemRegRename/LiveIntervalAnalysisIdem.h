#ifndef LLVM_LIVEINTERVALANALYSISIDEM_H
#define LLVM_LIVEINTERVALANALYSISIDEM_H

#include <llvm/PassSupport.h>
#include <llvm/CodeGen/MachineIdempotentRegions.h>
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include <algorithm>
#include <set>
#include <map>
#include <unordered_set>
#include "llvm/ADT/BitVector.h"

namespace llvm {
enum {
  LOAD,
  USE,
  DEF,
  STORE,
  NUM = 8
};

// forward declaration.
class RangeIterator;

class LiveRangeIdem {
public:
  unsigned start;
  unsigned end;
  LiveRangeIdem *next;
  LiveRangeIdem *prev;

public:
  LiveRangeIdem(unsigned _defIdx, unsigned _killIdx, LiveRangeIdem * _next, LiveRangeIdem *_prev) :
      start(_defIdx), end(_killIdx), next(_next), prev(_prev) {
    if (next) {
      next->prev = this;
    }
    if (prev) {
      prev->next = this;
    }
  }

  LiveRangeIdem() = default;

  bool contains(unsigned idx) {
    return start <= idx && idx <= end;
  }
  void print(llvm::raw_ostream &OS) {
    OS<<"["<<start<<", "<<end<<")";
  }
  void dump() { print(llvm::errs()); }

  /**
   * Determines the iterator position where this will interferes with
   * another {@link LiveRangeItem} {@code r} at the start of Live Range.
   *
   * Return the interfered iterator position which will equal to end()
   * if no interference exist.
   * @param r2
   * @return
   */
  RangeIterator intersectsAt(LiveRangeIdem *r2);
};

class UsePoint {
public:
  unsigned id;
  MachineOperand *mo;
  UsePoint(unsigned ID, MachineOperand *MO) : id(ID), mo(MO) {}

  bool operator <(const UsePoint &rhs) const {
    if (id < rhs.id) return true;
    if (id > rhs.id) return false;

    MachineInstr *mi = mo->getParent();
    assert(mi == rhs.mo->getParent());

    // Use first
    if (rhs.mo->isUse() && mo->isDef())
      return false;
    if (mo->isUse() && rhs.mo->isDef())
      return true;

    int moIdx = -1, rhsMOIdx = -1;
    for (int i = 0, e = mi->getNumOperands(); i < e; i++) {
      if (&mi->getOperand(i) == mo)
        moIdx = i;
      if (&mi->getOperand(i) == rhs.mo)
        rhsMOIdx = i;
    }

    assert(moIdx !=-1 && rhsMOIdx != -1);
    return moIdx < rhsMOIdx;
  }

  bool operator == (const UsePoint &rhs) const {
    return id == rhs.id && mo == rhs.mo;
  }
};

struct UsePointHasher {
  size_t operator()(const UsePoint &up) const {
    return (up.id << 13) ^ (size_t)up.mo;
  }
};

struct UsePointComparator {
  bool operator() (const UsePoint &lhs, const UsePoint &rhs) const {
    return lhs.id == rhs.id && lhs.mo == rhs.mo;
  }
};

class RangeIterator : public std::iterator<std::forward_iterator_tag, LiveRangeIdem*> {
private:
  LiveRangeIdem *cur;
public:
  explicit RangeIterator(LiveRangeIdem *_first) : cur(_first) {}
  RangeIterator() = default;

  RangeIterator &operator++() {
    cur = cur->next;
    return *this;
  }
  RangeIterator operator++(int) {
    RangeIterator res = *this;
    ++res;
    return res;
  }
  RangeIterator operator--(int) {
    RangeIterator res = *this;
    --res;
    return res;
  }

  RangeIterator operator--() {
    cur = cur->prev;
    return *this;
  }

  LiveRangeIdem* operator *() {
    return cur;
  }

  operator LiveRangeIdem*() { return cur; }

  bool operator ==(RangeIterator itr) {
    return cur == itr.cur;
  }
  bool operator !=(RangeIterator itr) { return !(*this == itr); }
  LiveRangeIdem *operator->() {
    assert(cur);
    return cur;
  }
};

class LiveIntervalAnalysisIdem;

class LiveIntervalIdem {
public:
  unsigned reg;
  /**
   * This is a backup of old physical register before renaming.
   * This field will be useful when choosing other physical register
   * as a replacement.
   */
  unsigned oldReg;
  LiveRangeIdem *first;
  LiveRangeIdem *last;
  typedef std::set<UsePoint> UsePointSet;
  UsePointSet usePoints;
  /**
   * Indicates the cost of spilling out this interval into memory.
   */
  unsigned costToSpill;

  /**
   * This flag is about to be used in IdemRegisterRenamer to check if
   * we should consider the anti-dependence caused by newly assigned register.
   */
  // bool fromLoad : 1;

  /**
   * This flag indicates there is a splitting at the beginning of this interval,
   * so we have to insert a move instruction as resolving dataflow.
   */
  bool insertedMove;
  LiveIntervalIdem *splitParent;
  std::vector<LiveIntervalIdem* > splitChildren;

  LiveIntervalIdem() : reg(0), first(nullptr),
                       last(nullptr),
                       usePoints(), costToSpill(0),
                       insertedMove(false),
                       splitParent(this),
                       splitChildren()/*,
                       fromLoad(false)*/ {}
  ~LiveIntervalIdem();

  UsePointSet &getUsePoint() { return usePoints; }

  void addRange(unsigned from, unsigned to);

  LiveRangeIdem *getLast() { return last; }
  void addUsePoint(unsigned numMI, MachineOperand *MO) {
    usePoints.insert(UsePoint(numMI, MO));
  }

  void print(llvm::raw_ostream &OS, const TargetRegisterInfo *tri);
  void dump(const TargetRegisterInfo *TRI = nullptr) { print(llvm::errs(), TRI); }
  bool isExpiredAt(unsigned pos) { return getLast()->end <= pos; }
  bool isLiveAt(unsigned pos);
  unsigned beginNumber() { return first->start; }
  unsigned beginNumber() const { return first->start; }
  unsigned endNumber() { return last->end; }
  unsigned endNumber() const { return last->end; }

  RangeIterator intersectAt(LiveIntervalIdem *li);
  bool intersects(LiveIntervalIdem *cur);

  typedef UsePointSet::iterator iterator;
  iterator usepoint_begin() { return usePoints.begin(); }
  iterator usepoint_end() { return usePoints.end(); }

  bool operator <(const LiveIntervalIdem &rhs) const {
    return beginNumber() < rhs.beginNumber();
  }

  bool empty() { return begin() == end(); }

  void split(LiveIntervalAnalysisIdem *li, MachineInstr *useMI, MachineInstr *copyMI, unsigned newReg);

  void setInsertedMove() { insertedMove = true; }

  LiveIntervalIdem *getSplitParent() {
    if (isSplitParent())
      return this;
    else
      return splitParent->getSplitParent();
  }

  bool isSplitParent() {
    return splitParent == this;
  }

  bool isSplitChildren() {
    return !isSplitParent();
  }

  bool hasSplitChildren() {
    return splitParent && !splitChildren.empty();
  }

  bool hasHoleBetween(unsigned from, unsigned to) {
    assert(from < to);
    if (to <= beginNumber() || from >= endNumber()) return false;
    LiveRangeIdem *temp = new LiveRangeIdem(from, to, nullptr, nullptr);
    return first->intersectsAt(temp) == end();
  }

  LiveIntervalIdem *getSplitChildBeforeOpId(unsigned id) {
    LiveIntervalIdem *parent = getSplitParent();
    LiveIntervalIdem *result = nullptr;

    assert(parent->hasSplitChildren() && "No split children available");
    size_t len = parent->splitChildren.size();
    for (int i = len - 1; i >= 0; --i) {
      LiveIntervalIdem *cur = parent->splitChildren[i];
      if (cur->endNumber() <= id && (!result || result->endNumber() <= cur->endNumber()))
        result = cur;
    }

    assert(result && "no split child found");
    return result;
  }

  LiveIntervalIdem *getSplitChildAtOpId(unsigned id) {
    LiveIntervalIdem *parent = getSplitParent();
    LiveIntervalIdem *result = nullptr;

    if (!parent->hasSplitChildren()) return nullptr;
    for (auto itr = parent->splitChildren.begin(),
             end = parent->splitChildren.end(); itr != end; ++itr) {
      LiveIntervalIdem *cur = *itr;
      if (cur->isLiveAt(id))
        result = cur;
    }
    return result;
  }

private:
  /**
   * Insert live range before the current range. It will merge range to be inserted with
   * adjacent range as necessary.
   * @param from
   * @param to
   * @param cur
   */
  void insertRangeBefore(unsigned from, unsigned to, LiveRangeIdem *&cur);

public:
  friend class RangeIterator;

  RangeIterator begin() { return RangeIterator{first}; }
  RangeIterator end() { return RangeIterator(); }
  const RangeIterator begin() const { return RangeIterator(first); }
  const RangeIterator end() const { return RangeIterator(); }
  void removeRange(unsigned from, unsigned to);
  RangeIterator upperBound(RangeIterator begin, RangeIterator end, unsigned key);
  void eraseRange(LiveRangeIdem *range) {
    if (!range) return;

    if (range->next != nullptr)
      range->next->prev = range->prev;
    if (range->prev != nullptr)
      range->prev->next = range->next;

    if (first == range)
      first = range->next;
    if (last == range)
      last = range->next;

    for (auto itr = usePoints.begin(), end = usePoints.end(); itr != end; ) {
      UsePoint up = *itr;
      if (up.id >= range->start && up.id < range->end)
        itr = usePoints.erase(itr);
      else
        ++itr;
    }

    delete range;
  }
  void resetStart(unsigned int usePos, unsigned int newStart);
  unsigned getUsePointAfter(unsigned int pos);
  unsigned getUsePointBefore(unsigned pos);
  int getFirstUse() {
    return usePoints.empty() ? -1 : usePoints.begin()->id;
  }
};

class LiveIntervalAnalysisIdem : public MachineFunctionPass {
public:
  /**
   * Map from instruction slot to corresponding machine instruction.
   */
  std::map<unsigned, MachineInstr*> idx2MI;
  /**
   * Map from machine instruction to it's number.
   */
  std::map<MachineInstr*, unsigned > mi2Idx;
  const TargetRegisterInfo* tri;
  /**
   * Map from original physical register to it's corresponding
   * live interval.
   */
  std::map<unsigned, LiveIntervalIdem*> intervals;
  std::vector<std::set<unsigned> > liveIns;
  std::vector<std::set<unsigned> > liveOuts;
  llvm::BitVector allocatableRegs;
  const MachineFunction *mf;
  MachineDominatorTree *dt;
  MachineLoopInfo *loopInfo;

  virtual void releaseMemory() override {
    idx2MI.clear();
    mi2Idx.clear();
    tri = nullptr;
    intervals.clear();
    liveIns.clear();
    liveOuts.clear();
    allocatableRegs.clear();
    mf = nullptr;
    dt = nullptr;
    loopInfo = nullptr;
  }
private:
  void addRegisterWithSubregs(std::set<unsigned> &set, unsigned reg) {
    set.insert(reg);
    if (!TargetRegisterInfo::isStackSlot(reg) &&
        TargetRegisterInfo::isPhysicalRegister(reg)) {
      for (const unsigned *r = tri->getSubRegisters(reg); *r; ++r)
        set.insert(*r);
    }
  }

  void computeLocalLiveSet(std::vector<std::set<unsigned> > &liveGens,
                           std::vector<std::set<unsigned> > &liveKills);

  void numberMachineInstr(std::vector<MachineBasicBlock*> &sequence);
  void computeGlobalLiveSet(std::vector<std::set<unsigned> > &liveIns,
                            std::vector<std::set<unsigned> > &liveOuts,
                            std::vector<std::set<unsigned> > &liveGens,
                            std::vector<std::set<unsigned> > &liveKills);
  void handleRegisterDef(unsigned reg, MachineOperand *mo, unsigned start, unsigned end);
  void buildIntervals(std::vector<MachineBasicBlock*> &sequence,
                      std::vector<std::set<unsigned> > &liveOuts);
  /**
   * Weight each live interval by computing the use number of live interval
   * according to it's loop nesting depth.
   */
  void weightLiveInterval();

public:
  static char ID;
  LiveIntervalAnalysisIdem() : MachineFunctionPass(ID) {
    initializeLiveIntervalAnalysisIdemPass(*PassRegistry::getPassRegistry());
  }

  virtual ~LiveIntervalAnalysisIdem() {
    releaseMemory();
  }

  const char *getPassName() const override {
    return "Live Interval computing for Register Renaming";
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override{
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  unsigned long getNumIntervals() const { return intervals.size(); }

  typedef std::map<unsigned, LiveIntervalIdem*>::iterator interval_iterator;
  typedef std::map<unsigned, LiveIntervalIdem*>::const_iterator const_interval_iterator;

  interval_iterator interval_begin() { return intervals.begin(); }
  const_interval_iterator interval_begin() const { return intervals.begin(); }

  interval_iterator interval_end() { return intervals.end(); }
  const_interval_iterator interval_end() const { return intervals.end(); }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineBasicBlock *getBlockAtId(unsigned pos) {
    for (const MachineBasicBlock &mbb : *mf) {
      unsigned from = getIndex(const_cast<MachineInstr*>(&mbb.front()));
      unsigned to = getIndex(const_cast<MachineInstr*>(&mbb.back()));
      if (from <= pos && pos < to + NUM)
        return const_cast<MachineBasicBlock *>(&mbb);
    }
    return nullptr;
  }

  unsigned getIndex(MachineInstr *mi) {
    assert(mi2Idx.count(mi));
    return mi2Idx[mi];
  }

  void setIndex(unsigned index, MachineInstr *mi) {
    assert(mi);
    mi2Idx[mi] = index;
    idx2MI[index] = mi;
  }

  MachineInstr* getMachineInstr(unsigned index) {
    assert(idx2MI.count(index));
    return idx2MI[index];
  }

  void insertOrCreateInterval(unsigned int reg, LiveIntervalIdem *pIdem);

  void dump();
  void removeInterval(LiveIntervalIdem *pIdem);
  /**
   * Update the live interval of the specified reg used at the position specified
   * by usePos with the new start index.
   * @param oldReg
   * @param usePos
   * @param useMI
   * @param copyMI
   */
  void resetLiveIntervalStart(unsigned oldReg,
                              unsigned usePos,
                              MachineOperand *mo);

  void buildIntervalForRegister(unsigned reg, MachineOperand *mo);

  LiveIntervalIdem* split(unsigned splitPos, LiveIntervalIdem *it);

  bool isBlockBegin(unsigned pos) {
    auto itr = std::find_if(mi2Idx.begin(), mi2Idx.end(), [&](const std::pair<MachineInstr*, unsigned> &pair) {
      return pair.second == pos;
    });
    assert(itr != mi2Idx.end());
    return itr->first == &itr->first->getParent()->front();
  }


  void computeCostToSpill(LiveIntervalIdem *it) {
    // Weight each use point by it's loop nesting deepth.
    unsigned cost = 0;
    for (auto &up : it->usePoints) {
      MachineBasicBlock *mbb = up.mo->getParent()->getParent();
      if (MachineLoop *ml = loopInfo->getLoopFor(mbb)) {
        cost += 10 * ml->getLoopDepth();
      } else
        cost += 1;
    }
    it->costToSpill = cost;
  }
};
}

#endif //LLVM_LIVEINTERVALANALYSISIDEM_H
