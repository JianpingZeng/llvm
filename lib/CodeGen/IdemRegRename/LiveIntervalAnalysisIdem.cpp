#include <deque>
#include <llvm/ADT/SetOperations.h>
#include <llvm/Support/Debug.h>
#include "LiveIntervalAnalysisIdem.h"
#include "IdemUtil.h"

using namespace llvm;

RangeIterator LiveRangeIdem::intersectsAt(LiveRangeIdem *r2) {
  assert(r2);
  RangeIterator itr1(this), itr2(r2), end(0);
  while (true) {
    if (itr1->start < itr2->start) {
      if (itr1->end <= itr2->start) {
        ++itr1;
        if (itr1 == end)
          return end;
      } else
        return itr2;
    } else {
      if (itr1->start == itr2->start)
        return itr1;
      // Otherwise, r1.start > r2.start <--> r2.start < r1.start
      if (itr2->end <= itr1->start) {
        ++itr2;
        if (itr2 == end)
          return end;
      } else
        return itr1;
    }
  }
}

LiveIntervalIdem::~LiveIntervalIdem() {
  if (first) {
    LiveRangeIdem *cur = first;
    while (cur) {
      LiveRangeIdem *next = cur->next;
      delete cur;
      cur = next;
    }
  }
}

void LiveIntervalIdem::addRange(unsigned from, unsigned to) {
  assert(from <= to && "Invalid range!");
  if (first == nullptr || to <= first->end)
    insertRangeBefore(from, to, first);
  else {
    LiveRangeIdem **r = &first, *prevPos = nullptr;
    while (*r != nullptr) {
      if (to > (*r)->end) {
        prevPos = *r;
        r = &(*r)->next;
      } else
        break;
    }

    // insert the range in the tailing of linked list.
    if (!*r) {
      last = new LiveRangeIdem(from, to, nullptr, prevPos);
    } else
      insertRangeBefore(from, to, *r);
  }
}

void LiveIntervalIdem::print(llvm::raw_ostream &OS, const TargetRegisterInfo *tri) {
  OS << (TargetRegisterInfo::isPhysicalRegister(reg) && tri ?
         tri->getName(reg) : ("%vreg" + reg));
  LiveRangeIdem *r = first;
  while (r != nullptr) {
    r->dump();
    OS << ",";
    r = r->next;
  }

  OS << " Use points: [";
  unsigned long i = 0, size = usePoints.size();
  for (UsePoint up : usePoints) {
    OS << up.id;
    if (i < size - 1)
      OS << ",";
    ++i;
  }
  OS << "]\n";
}

bool LiveIntervalIdem::isLiveAt(unsigned pos) {
  if (pos <= first->start || pos >= last->end)
    return false;

  LiveRangeIdem *itr = first;
  while (itr != nullptr) {
    if (itr->contains(pos))
      return true;
    itr = itr->next;
  }
  return false;
}

bool LiveIntervalIdem::intersects(LiveIntervalIdem *cur) {
  assert(cur);
  if (cur->beginNumber() > endNumber())
    return false;

  return intersectAt(cur) != end();
}
RangeIterator LiveIntervalIdem::intersectAt(LiveIntervalIdem *li) {
  return first->intersectsAt(li->first);
}

void LiveIntervalIdem::insertRangeBefore(unsigned from, unsigned to, LiveRangeIdem *&cur) {
  assert((cur == nullptr || to <= cur->end) && "Not inserting at begining of interval");
  if (!cur) {
    last = cur = new LiveRangeIdem(from, to, nullptr, last);
    return;
  }
  if (cur->start <= to) {
    assert(cur != nullptr && "First range must not be EndMarker");
    cur->start = std::min(from, cur->start);
    cur->end = std::max(to, cur->end);
  } else {
    cur = new LiveRangeIdem(from, to, cur, cur->prev);
  }
}

RangeIterator LiveIntervalIdem::upperBound(RangeIterator begin, RangeIterator end,
                                           unsigned key) {
  for (; begin != end; ++begin) {
    if (begin->start > key)
      return begin;
  }
  return end;
}

void LiveIntervalIdem::removeRange(unsigned from, unsigned to) {
  if (to <= beginNumber() || from >= endNumber())
    return;
  if (from < beginNumber())
    from = beginNumber();
  if (to > endNumber())
    to = endNumber();

  RangeIterator upper = upperBound(begin(), end(), from);
  if (upper == end())
    upper = RangeIterator(last);
  else
    --upper;

  assert(upper->contains(to - 1) && "LiveRangeIdem is not entirely in interval!");
  for (auto itr = usePoints.begin(), end = usepoint_end(); itr != end; ) {
    if (from <= itr->id && itr->id < to)
      itr = usePoints.erase(itr);
    else
      ++itr;
  }

  if (upper->start == from) {
    if (upper->end == to) {
      eraseRange(upper);
    } else
      upper->start = to;
    return;
  }

  if (upper->end == to) {
    upper->end = from;
    return;
  }

  unsigned oldEnd = upper->end;
  upper->end = from;
  LiveRangeIdem *pos = *(++upper);
  insertRangeBefore(to, oldEnd, pos);
}

void LiveIntervalIdem::resetStart(unsigned int usePos, unsigned int newStart) {
  if (usePos < beginNumber() || usePos >= endNumber())
    return;

  RangeIterator cur = upperBound(begin(), end(), usePos);
  --cur;
  assert(cur != end() && cur->contains(usePos) &&
      "The use position is not within live range!");
  // Reset the start position of the specified live range
  if (cur->start < newStart) {
    // remove some use point
    for (auto itr = usepoint_begin(), end = usepoint_end(); itr != end; ++itr) {
      if (itr->id >= cur->start && itr->id < newStart)
        usePoints.erase(itr);
    }
    cur->start = newStart;
  }
}

void LiveIntervalIdem::split(LiveIntervalAnalysisIdem *li,
                             MachineInstr *useMI,
                             MachineInstr *copyMI,
                             unsigned newReg) {
  unsigned useIndex = li->getIndex(useMI);
  unsigned copyIndex = li->getIndex(copyMI);
  LiveIntervalIdem *newInterval = new LiveIntervalIdem;
  newInterval->reg = newReg;
  RangeIterator cur = upperBound(begin(), end(), useIndex);
  if (cur == end())
    cur = RangeIterator(last);
  else
    --cur;
  assert(cur != end());

  if (copyIndex > cur->end)
    std::swap(copyIndex, cur->end);

  newInterval->addRange(copyIndex, cur->end);
  newInterval->addUsePoint(copyIndex, &copyMI->getOperand(0));

  addUsePoint(copyIndex, &copyMI->getOperand(1));
  for (auto itr = usepoint_begin(), end = usepoint_end(); itr != end; ) {
    if (itr->id >= cur->start && itr->id < cur->end) {
      newInterval->addUsePoint(itr->id, itr->mo);
      itr = usePoints.erase(itr);
    }
    else
      ++itr;
  }
  cur->end = copyIndex + 1;
  RangeIterator begin = cur;

  cur->next = nullptr;
  last = cur;

  ++begin;
  newInterval->first->next = begin;
  if (begin)
    begin->prev = newInterval->first;

  newInterval->last = begin;
  for (; begin != end(); ++begin) {
    for (auto itr = usepoint_begin(), end = usepoint_end(); itr != end; ) {
      if (itr->id >= begin->start && itr->id < begin->end) {
        newInterval->addUsePoint(itr->id, itr->mo);
        itr = usePoints.erase(itr);
      }
      else
        ++itr;
    }
    newInterval->last = begin;
  }
}

unsigned LiveIntervalIdem::getUsePointAfter(unsigned int pos) {
  UsePoint *up = 0;
  for (auto itr : usePoints) {
    if (itr.id >= pos) {
      if (!up || itr.id < up->id)
        up = &itr;
    }
  }
  return up ? up->id : UINT32_MAX;
}

unsigned LiveIntervalIdem::getUsePointBefore(unsigned pos) {
  assert(pos >= beginNumber());

  for (auto itr = usePoints.rbegin(), end = usePoints.rend(); itr != end; ++itr) {
    if (itr->id < pos)
      return itr->id;
  }
  return pos - 1;
}

char LiveIntervalAnalysisIdem::ID = 0;
INITIALIZE_PASS_BEGIN(LiveIntervalAnalysisIdem, "live-interval-idem",
                      "Live Interval computing for Register Renaming", false, false)
  INITIALIZE_PASS_DEPENDENCY(LiveVariables)
  INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
  INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(LiveIntervalAnalysisIdem, "live-interval-idem",
                    "Live Interval computing for Register Renaming", false, false)

void LiveIntervalAnalysisIdem::computeLocalLiveSet(std::vector<std::set<unsigned> > &liveGens,
                                                   std::vector<std::set<unsigned> > &liveKills) {
  for (auto &mbb : *mf) {
    auto itr = mbb.begin();
    auto end = mbb.end();
    for (; itr != end; ++itr) {
      const MachineInstr *mi = &*itr;
      for (int j = mi->getNumOperands() - 1; j >= 0; --j) {
        const MachineOperand &mo = mi->getOperand(j);
        if (!mo.isReg() || !mo.getReg() ||
            // We don't count such special registers
            tri->isNotCountedAsLiveness(mo.getReg()))
          continue;

        unsigned reg = mo.getReg();
        if (mo.isUse() && !liveKills[mbb.getNumber()].count(reg))
          addRegisterWithSubregs(liveGens[mbb.getNumber()], reg);
        else
          addRegisterWithSubregs(liveKills[mbb.getNumber()], reg);
      }
    }
  }
}

void LiveIntervalAnalysisIdem::numberMachineInstr(std::vector<MachineBasicBlock *> &sequence) {
  if (sequence.empty())
    return;
  unsigned totalMIs = 0;
  for (auto mbb : sequence)
    totalMIs += mbb->size();

  idx2MI.clear();
  mi2Idx.clear();

  // starts from 4 in the case of we will insert a interval before first instr.
  unsigned index = NUM;
  for (auto mbb : sequence) {
    auto mi = mbb->instr_begin();
    auto end = mbb->instr_end();
    for (; mi != end; ++mi) {
      mi2Idx[&*mi] = index;
      idx2MI[index] = &*mi;
      index += NUM;
    }
  }
}

void LiveIntervalAnalysisIdem::computeGlobalLiveSet(std::vector<std::set<unsigned> > &liveIns,
                                                    std::vector<std::set<unsigned> > &liveOuts,
                                                    std::vector<std::set<unsigned> > &liveGens,
                                                    std::vector<std::set<unsigned> > &liveKills) {
  bool changed;
  auto RegSetEq = [](std::set<unsigned> &lhs, std::set<unsigned> &rhs) {
    if (lhs.size() != rhs.size())
      return false;
    auto itr = lhs.begin(), end = lhs.end();
    auto rhsItr = rhs.begin();
    for (; itr != end; ++itr, ++rhsItr) {
      if (*itr != *rhsItr)
        return false;
    }
    return true;
  };

  do {
    changed = false;
    for (auto &mbb : *mf) {
      auto out = std::set<unsigned>();
      if (!mbb.succ_empty()) {
        for (auto succ = mbb.succ_begin(), succEnd = mbb.succ_end(); succ != succEnd; ++succ)
          set_union(out, liveIns[(*succ)->getNumber()]);
      }

      set_union(out, liveOuts[mbb.getNumber()]);
      bool localChanged = !RegSetEq(out, liveOuts[mbb.getNumber()]);
      if (localChanged) {
        changed = true;
        liveOuts[mbb.getNumber()] = out;
      }

      auto in = out;
      set_subtract(in, liveKills[mbb.getNumber()]);
      set_union(in, liveGens[mbb.getNumber()]);
      localChanged = !RegSetEq(in, liveIns[mbb.getNumber()]);
      if (localChanged) {
        changed = true;
        liveIns[mbb.getNumber()] = in;
      }
    }
  } while (changed);
}

void LiveIntervalAnalysisIdem::handleRegisterDef(unsigned reg,
                                                 MachineOperand *mo,
                                                 unsigned start,
                                                 unsigned end) {
  LiveIntervalIdem *&li = intervals[reg];
  if (!li) {
    li = new LiveIntervalIdem();
    li->reg = reg;
  }
  assert(li && "must be no null");
  if (mo->isDead()) {
    li->addRange(start, start + 1);
    li->addUsePoint(start, mo);
  } else {
    LiveRangeIdem *&lr = li->first;
    if (!lr)
      li->addRange(start, end);
    else
      lr->start = start;
    li->addUsePoint(start, mo);
  }
}

void LiveIntervalAnalysisIdem::buildIntervals(std::vector<MachineBasicBlock *> &sequence,
                                              std::vector<std::set<unsigned> > &liveOuts) {
  intervals.clear();
  auto itr = sequence.rbegin();
  auto end = sequence.rend();
  for (; itr != end; ++itr) {
    MachineBasicBlock *mbb = *itr;
    if (mbb->empty())
      continue;

    assert(mi2Idx.count(&mbb->front()));
    assert(mi2Idx.count(&mbb->back()));
    unsigned blockFrom = mi2Idx[&mbb->front()];
    unsigned blockTo = mi2Idx[&mbb->back()] + NUM;
    std::set<unsigned> &set = liveOuts[mbb->getNumber()];
    for (unsigned reg : set) {
      LiveIntervalIdem *&li = intervals[reg];
      if (!li) {
        li = new LiveIntervalIdem();
        li->reg = reg;
      }

      li->addRange(blockFrom, blockTo);
    }

    for (auto mi = mbb->instr_rbegin(), miEnd = mbb->instr_rend(); mi != miEnd; ++mi) {
      unsigned num = mi2Idx[&*mi];
      std::vector<MachineOperand *> uses, defs;
      for (unsigned moIdx = 0, sz = mi->getNumOperands(); moIdx < sz; ++moIdx) {
        MachineOperand &mo = mi->getOperand(moIdx);
        if (mo.isReg() && mo.getReg()) {
          // const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(mo.getReg());
          allocatableRegs = tri->getAllocatableSet(*mf);
          // skip unallocable register.
          if (!TargetRegisterInfo::isPhysicalRegister(mo.getReg()) ||
              !allocatableRegs[mo.getReg()])
            continue;
          if (mo.isDef())
            defs.push_back(&mo);
          else if (mo.isUse())
            uses.push_back(&mo);
        }
      }

      // handle defined registers.
      for (MachineOperand *op : defs) {
        unsigned reg = op->getReg();
        handleRegisterDef(reg, op, num, blockTo);
        if (TargetRegisterInfo::isPhysicalRegister(reg)) {
          const unsigned *subregs = tri->getSubRegisters(reg);
          if (subregs)
            for (; *subregs; ++subregs)
              if (!mi->modifiesRegister(*subregs, tri))
                handleRegisterDef(*subregs, op, num, blockTo);
        }
      }

      // handle use registers.
      for (MachineOperand *op : uses) {
        unsigned reg = op->getReg();
        LiveIntervalIdem *&li = intervals[reg];
        if (!li) {
          li = new LiveIntervalIdem();
          li->reg = reg;
        }

        assert(li);
        li->addRange(blockFrom, num);
        // extends the use to cross current instruction.
        if (li->first->end == num && li->first->start > 0)
          --li->first->start;

        li->addUsePoint(num, op);
      }
    }
  }
}

bool LiveIntervalAnalysisIdem::runOnMachineFunction(MachineFunction &MF) {
  mf = &MF;
  tri = MF.getTarget().getRegisterInfo();
  unsigned size = MF.getNumBlockIDs();
  dt = getAnalysisIfAvailable<MachineDominatorTree>();
  loopInfo = getAnalysisIfAvailable<MachineLoopInfo>();
  assert(dt);

  // Step #1: compute block order.
  std::vector<MachineBasicBlock *> sequence;
  computeReversePostOrder(MF, sequence);

  // Step#2: computes local data flow information.
  std::vector<std::set<unsigned> > liveGens(size);
  std::vector<std::set<unsigned> > liveKills(size);
  computeLocalLiveSet(liveGens, liveKills);

  // Step#3: compute global live set.
  liveIns.resize(size);
  liveOuts.resize(size);
  computeGlobalLiveSet(liveIns, liveOuts, liveGens, liveKills);

  // Step $4: number the machine instrs.
  numberMachineInstr(sequence);

  // Step#5: build intervals.
  buildIntervals(sequence, liveOuts);

  // Step#6:
  weightLiveInterval();

  // Dump some useful information for it to review the correctness
  // of this transformation.
  IDEM_DEBUG(dump(););
  return false;
}

void LiveIntervalAnalysisIdem::insertOrCreateInterval(unsigned int reg,
                                                      LiveIntervalIdem *pIdem) {
  if (intervals.count(reg)) {
    auto itr = pIdem->begin();
    auto end = pIdem->end();
    for (; itr != end; ++itr)
      intervals[reg]->addRange(itr->start, itr->end);
    // accumulate cost to be spilled.
    if (pIdem->costToSpill == UINT32_MAX)
      // avoiding overflow
      intervals[reg]->costToSpill = UINT32_MAX;
    else
      intervals[reg]->costToSpill += pIdem->costToSpill;
    // add use points to the interval.
    intervals[reg]->usePoints.insert(pIdem->usepoint_begin(), pIdem->usepoint_end());
  } else {
    intervals.insert(std::pair<unsigned, LiveIntervalIdem *>(reg, pIdem));
  }
}

void LiveIntervalAnalysisIdem::weightLiveInterval() {
  // loop over all live intervals to compute spill cost.
  auto itr = interval_begin();
  auto end = interval_end();
  for (; itr != end; ++itr) {
    // Weight each use point by it's loop nesting deepth.
    computeCostToSpill(itr->second);
  }
}

void LiveIntervalAnalysisIdem::dump() {
  llvm::errs() << "\nMachine instruction and Slot: \n";
  for (auto &mbb : *mf) {
    auto mi = mbb.instr_begin();
    auto end = mbb.instr_end();
    for (; mi != end; ++mi) {
      llvm::errs() << mi2Idx[const_cast<MachineInstr *>(&*mi)] << ":";
      mi->dump();
    }
  }

  llvm::errs() << "\nLive Interval for Idempotence:\n";
  auto li = interval_begin();
  auto end = interval_end();
  for (; li != end; ++li) {
    li->second->dump(mf->getTarget().getRegisterInfo());
    llvm::errs() << "\n";
  }
}

void LiveIntervalAnalysisIdem::removeInterval(LiveIntervalIdem *pIdem) {
  for (auto itr = interval_begin(), end = interval_end(); itr != end; ++itr) {
    if (itr->first == pIdem->reg) {
      for (auto r = pIdem->begin(), e = pIdem->end(); r != e; ++r)
        itr->second->removeRange(r->start, r->end);

      if (itr->second->empty())
        intervals.erase(itr);
    }
  }
}

void LiveIntervalAnalysisIdem::resetLiveIntervalStart(unsigned int oldReg,
                                                      unsigned int usePos,
                                                      MachineOperand *mo) {
  // If there is no live interval associated with the specified oldReg,
  // terminates immediately.
  if (!intervals[oldReg]) return;

  LiveIntervalIdem *interval = intervals[oldReg];
  unsigned newStart = getIndex(mo->getParent());
  interval->resetStart(usePos, newStart);
  interval->addUsePoint(newStart, mo);
}

void LiveIntervalAnalysisIdem::buildIntervalForRegister(unsigned reg,
                                                        MachineOperand *mo) {
  assert(TargetRegisterInfo::isPhysicalRegister(reg));
  LiveIntervalIdem *&interval = intervals[reg];
  if (!interval) {
    interval = new LiveIntervalIdem;
    interval->reg = reg;
  }
  MachineBasicBlock *mbb = mo->getParent()->getParent();
  unsigned blockBegin = getIndex(&mbb->front());
  unsigned to = getIndex(mo->getParent()) + 1;
  interval->addRange(blockBegin, to);
  interval->addUsePoint(to-1, mo);

  // Add Live Range for each MBB whose live out set contains the reg
  std::vector<MachineBasicBlock*> worklist;
  std::set<MachineBasicBlock*> visited;
  worklist.push_back(mbb);
  while (!worklist.empty()) {
    auto cur = worklist.back();
    worklist.pop_back();
    visited.insert(cur);

    bool shouldForward = true;
    MachineInstr *def = 0;
    MachineOperand *defMO = 0;
    for (auto itr = cur->rbegin(), end = cur->rend(); itr != end && shouldForward; ++itr) {
      for (unsigned i = 0, e = itr->getNumOperands(); i < e; i++) {
        auto &mo = itr->getOperand(i);
        if (mo.isReg() && mo.getReg() && mo.getReg() == reg) {
          shouldForward = false;
          def = &*itr;
          defMO = &mo;
          break;
        }
      }
    }
    if (shouldForward) {
      interval->addRange(getIndex(&cur->front()), getIndex(&cur->back()) + 1);
      std::vector<MachineBasicBlock*> buf;
      buf.assign(cur->pred_begin(), cur->pred_end());
      std::for_each(buf.rbegin(), buf.rend(), [&](MachineBasicBlock *pred) {
        if (!visited.count(pred))
          worklist.push_back(pred);
      });
    }
    else {
      interval->addRange(getIndex(def), getIndex(&cur->back()) + 1);
      interval->addUsePoint(getIndex(def), defMO);
    }
  }

  // Weight cost of the live interval to spill
  // Weight each use point by it's loop nesting deepth.
  unsigned &cost = interval->costToSpill;
  for (auto &up : interval->usePoints) {
    MachineBasicBlock *mbb = up.mo->getParent()->getParent();
    if (MachineLoop *ml = loopInfo->getLoopFor(mbb)) {
      cost += 10 * ml->getLoopDepth();
    } else
      cost += 1;
  }
}

LiveIntervalIdem* LiveIntervalAnalysisIdem::split(unsigned splitPos, LiveIntervalIdem *it) {
  LiveIntervalIdem *child = new LiveIntervalIdem;
  child->reg = it->reg;
  child->oldReg = it->oldReg;
  child->splitParent = it->getSplitParent();
  it->getSplitParent()->splitChildren.push_back(child);

  LiveRangeIdem *cur = it->first, *prev = nullptr;
  while (cur && cur->end <= splitPos) {
    prev = cur;
    cur = cur->next;
  }

  assert(cur && "Split position after the end number of live interval!");
  if (cur->start < splitPos) {
    /*
     *     splitPos
     *       |
     * |----------------|
     * ^                ^
     * cur.from      cur.to
     */
    child->first = new LiveRangeIdem(splitPos, cur->end, cur->next, nullptr);
    if (it->last == cur)
      child->last = child->first;
    else
      child->last = it->last;

    cur->end = splitPos;
    cur->next = nullptr;
    it->last = cur;
  }
  else {
    /*
     * splitPos
     * |
     * |----------------|
     * ^                ^
     * cur.from      cur.to
     * where, splitPos <= cur.from
     */
    child->first = cur;
    cur->prev  = nullptr;

    if (it->last == cur)
      child->last = cur;
    else
      child->last = it->last;

    assert(prev && "Split position before begin number!");

    prev->next = nullptr;
    prev->end = splitPos;
    it->last = prev;
  }

  // Split the use points
  std::set<UsePoint> childUsePoints;
  for (auto itr = it->usepoint_begin(), end = it->usepoint_end(); itr != end; ) {
    UsePoint up = *itr;
    if (up.id >= splitPos) {
      childUsePoints.insert(up);
      itr = it->usePoints.erase(itr);
    } else
      ++itr;
  }
  child->usePoints = childUsePoints;
  return child;
}