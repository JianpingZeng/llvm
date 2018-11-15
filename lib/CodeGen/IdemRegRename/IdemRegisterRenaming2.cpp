//===----- IdemRegisterRenaming.cpp - Register Renaming after RA ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-renaming"

#include "IdemRegisterRenaming2.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

INITIALIZE_PASS_BEGIN(IdemRegisterRenamer, "reg-renaming",
                      "Register Renaming for Idempotence", false, false)
  INITIALIZE_PASS_DEPENDENCY(LiveIntervalAnalysisIdem)
  INITIALIZE_PASS_DEPENDENCY(MachineIdempotentRegions)
  INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
  INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
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
        if (!mo.isReg() || !mo.getReg() || !partialEquals(mo.getReg(), reg))
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
  if (!mbb) return;
  visited.insert(mbb);

  for (auto itr = idem; itr != end; ++itr) {
    if (tii->isIdemBoundary(itr))
      return;

    // iterate over operands from right to left, which means cope with
    // use oprs firstly, then def regs.
    for (int i = itr->getNumOperands() - 1; i >= 0; i--) {
      auto mo = itr->getOperand(i);
      if (!mo.isReg() || !mo.getReg())
        continue;

      if (mo.isUse() && mo.getReg() == reg)
        uses.emplace_back(itr, i);
      else if (mo.isDef() && partialEquals(mo.getReg(), reg)) {
        if (!defs.empty() && defs.begin()->mi->getOperand(defs.begin()->index).getReg() != mo.getReg())
          return;

        defs.emplace_back(itr, i);

        ++itr;
        bool ends = false;
        std::set<MachineBasicBlock *> tmpVisited;
        useDefChainEnds(reg, tmpVisited, itr, end, mbb, ends);
        if (ends) {
          // Construct anti-dependencies according uses and defs set.
          AntiDeps ad(reg, uses, defs);
          if (std::find(antiDeps.begin(), antiDeps.end(), ad) == antiDeps.end())
            antiDeps.push_back(ad);
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

void IdemRegisterRenamer::countRegisterRaiseAntiDepsInLoop(const MachineBasicBlock::iterator &idem,
                                                           const MachineBasicBlock::iterator &end,
                                                           MachineBasicBlock *mbb,
                                                           DenseSet<unsigned> &unallocableRegs) {
  if (!mbb)
    return;

  std::set<MachineBasicBlock*> visited;
  std::vector<MachineBasicBlock*> worklist;
  worklist.push_back(mbb);
  while (!worklist.empty()) {
    auto cur = worklist.back();
    worklist.pop_back();

    if (!visited.insert(cur).second)
      continue;

    bool shouldWalkSucc = true;
    if (!cur->empty()) {
      for (auto &mi : *cur) {
        if (tii->isIdemBoundary(&mi)) {
          shouldWalkSucc = false;
          break;
        }
        for (unsigned i = 0, e = mi.getNumOperands(); i < e; i++) {
          auto mo = mi.getOperand(i);
          if (mo.isReg() && mo.isDef() &&
              TargetRegisterInfo::isPhysicalRegister(mo.getReg()) &&
              !reservedRegs[mo.getReg()]) {
            addRegisterWithSubregs(unallocableRegs, mo.getReg());
            addRegisterWithSuperRegs(unallocableRegs, mo.getReg());
          }
        }
      }
    }
    if (shouldWalkSucc) {
      for (auto succ = cur->succ_begin(), succEnd = cur->succ_end(); succ != succEnd; ++succ)
        if (!visited.count(*succ))
          worklist.push_back(*succ);
    }
  }
}

void IdemRegisterRenamer::computeAntiDepsInLoop(unsigned reg,
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
      if (!mo.isReg() || !mo.getReg())
        continue;

      if (mo.isUse() && mo.getReg() == reg)
        uses.emplace_back(itr, i);
      else if (mo.isDef() && partialEquals(mo.getReg(), reg)) {
        if (!defs.empty() && defs.begin()->mi->getOperand(defs.begin()->index).getReg() != mo.getReg())
          return;

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

void IdemRegisterRenamer::gatherAntiDeps(MachineInstr *idem) {
  auto liveIns = gather->getIdemLiveIns(idem);
  if (liveIns.empty())
    return;

  auto begin = ++MachineBasicBlock::iterator(idem);
  /*if (idem->getParent()->getName() == "if.else.i46.i") {
    std::for_each(liveIns.begin(), liveIns.end(), [=](unsigned r) {
      llvm::dbgs()<<tri->getName(r)<<",";
    });
    llvm::errs()<<"\n";
  }*/
  for (auto reg : liveIns) {
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
    size_t before = antiDeps.size();
    gatherAntiDeps(idem);
    size_t after = antiDeps.size();
    region2NumAntiDeps[idem] = after - before;
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

/*static void getDefUses(MachineInstr *mi,
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
}*/
/**
 * Checks if it is legal to replace the oldReg with newReg. If so, return true.
 * @param newReg
 * @param oldReg
 * @return
 */
bool IdemRegisterRenamer::legalToReplace(unsigned newReg, unsigned oldReg) {
  if (newReg == 0 || TargetRegisterInfo::isVirtualRegister(newReg) ||
      reservedRegs[newReg])
    return false;

  for (unsigned i = 0, e = tri->getNumRegClasses(); i < e; i++) {
    auto rc = tri->getRegClass(i);
    if (tri->canUsedToReplace(newReg) && rc->contains(newReg) && rc->contains(oldReg))
      return true;
  }
  return false;
}

bool IdemRegisterRenamer::regIsRegForRC(unsigned newReg, const TargetRegisterClass *rc) {
  if (newReg == 0 || TargetRegisterInfo::isVirtualRegister(newReg) ||
      reservedRegs[newReg] || !tri->canUsedToReplace(newReg))
    return false;

  for (unsigned i = 0, e = tri->getNumRegClasses(); i < e; i++) {
    if (tri->getRegClass(i)->contains(newReg) && (rc->hasSubClassEq(tri->getRegClass(i)) ||
        tri->getRegClass(i)->hasSubClassEq(rc)))
      return true;
  }
  return false;
}

/*LLVM_ATTRIBUTE_UNUSED void IdemRegisterRenamer::filterUnavailableRegs(MachineOperand *use,
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
}*/

unsigned IdemRegisterRenamer::tryChooseFreeRegister(LiveIntervalIdem &interval,
                                                    int useReg,
                                                    BitVector &allocSet) {
  IDEM_DEBUG(llvm::errs() << "Interval for move instr: ";
                 interval.dump(tri);
                 llvm::errs() << "\n";);

  for (int physReg = allocSet.find_first(); physReg > 0; physReg = allocSet.find_next(physReg)) {
    if (li->intervals.count(physReg)) {
      LiveIntervalIdem *itrv = li->intervals[physReg];

      IDEM_DEBUG(llvm::errs() << "Candidate interval: ";
                     itrv->dump(tri);
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

bool IdemRegisterRenamer::getSpilledSubLiveInterval(LiveIntervalIdem *interval,
                                                    std::vector<LiveIntervalIdem *> &spilledItrs) {

  // We need to assign the same register to the use point in the two address instruction.
  std::vector<LiveIntervalIdem *> buf;
  std::set<std::pair<MachineInstr *, unsigned>> twoAddrInstrs;

  for (auto begin = interval->usepoint_begin(),
           end = interval->usepoint_end(); begin != end; ++begin) {
    if (isTwoAddressInstr(begin->mo->getParent(), interval->reg)) {
      twoAddrInstrs.insert(std::make_pair(begin->mo->getParent(), begin->id));
    }
    else {
      LiveIntervalIdem *verifyLI = new LiveIntervalIdem;
      verifyLI->addUsePoint(begin->id, begin->mo);
      unsigned from, to;
      if (begin->mo->isUse()) {
        to = li->getIndex(begin->mo->getParent());
        from = to - 2;
      } else {
        from = li->getIndex(begin->mo->getParent());
        to = from + 2;
      }

      verifyLI->addRange(from, to);

      // Keep the old register for choosing an appropriate register class when performing spilling
      verifyLI->reg = interval->reg;
      // tells compiler not to evict this spilling interval.
      verifyLI->costToSpill = UINT32_MAX;
      buf.push_back(verifyLI);
    }
  }

  // Delete the targetInter from LiveIntervalAnalysisIdem
  li->removeInterval(interval);

  for (std::pair<MachineInstr*, unsigned > pair : twoAddrInstrs) {
    MachineInstr *mi = pair.first;
    unsigned index = pair.second;
    unsigned from, to;
    from = li->getIndex(mi) - 2;
    to = from + 4;
    LiveIntervalIdem *verifyLI = new LiveIntervalIdem;
    for (unsigned i = 0, e = mi->getNumOperands(); i < e; i++) {
      MachineOperand& mo = mi->getOperand(i);
      if (mo.isReg() && mo.getReg() == interval->reg) {
        verifyLI->addUsePoint(index, &mo);
      }
    }

    for (auto itr = verifyLI->usepoint_begin(), end = verifyLI->usepoint_end(); itr != end; ++itr)
      assert(itr->mo->isReg());

    verifyLI->addRange(from, to);
    verifyLI->reg = interval->reg;
    verifyLI->costToSpill = UINT32_MAX;
    // verifyLI->fromLoad = true;
    buf.push_back(verifyLI);
  }

  spilledItrs.insert(spilledItrs.end(), buf.begin(), buf.end());
  return true;
}

void IdemRegisterRenamer::getAllocableRegs(unsigned useReg, std::set<unsigned> &allocables) {
  for (unsigned i = 0, e = tri->getNumRegClasses(); i < e; i++) {
    auto rc = tri->getRegClass(i);
    if (rc->contains(useReg))
      for (auto reg : rc->getRawAllocationOrder(*mf))
        if (!reservedRegs[reg])
          allocables.insert(reg);
  }
}

// for a group of sub live interval caused by splitting the original live interval.
// all of sub intervals in the same group have to be assigned a same frame index.
/*static std::map<LiveIntervalIdem *, unsigned> intervalGrpId;
static std::map<unsigned, int> grp2FrameIndex;

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
}*/

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

#if 0
void IdemRegisterRenamer::insertSpillingCodeForInterval(LiveIntervalIdem *spilledItr) {
  int frameIndex;

  for (auto itr = spilledItr->usepoint_begin(), end = spilledItr->usepoint_end(); itr != end; ++itr) {
    MachineOperand *mo = itr->mo;
    MachineInstr *mi = mo->getParent();
    assert(mo->isReg());
    unsigned oldReg = mo->getReg();
    unsigned usedReg = spilledItr->reg;
    mo->setReg(usedReg);

    const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(oldReg);
    if (hasFrameSlot(spilledItr))
      frameIndex = getFrameIndex(spilledItr);
    else {
      frameIndex = mfi->CreateSpillStackObject(rc->getSize(), rc->getAlignment());
      setFrameIndex(spilledItr, frameIndex);
    }

    if (mo->isDef() && !mo->isDead()) {

      auto st = getNextMI(mi);
      tii->storeRegToStackSlot(*mi->getParent(), st,
                               usedReg, false, frameIndex, rc, tri);

      st = getNextMI(mi);
      li->mi2Idx[st] = li->getIndex(mi) + 1;

    } else if (mo->isUse() && !mo->isKill()) {
      assert(frameIndex != INT_MIN);
      tii->loadRegFromStackSlot(*mi->getParent(), mi, usedReg, frameIndex, rc, tri);
      auto ld = getPrevMI(mi);
      li->mi2Idx[ld] = li->getIndex(mi) - 1;

      std::vector<IdempotentRegion*> regions;
      mir->getRegionsContaining(*mi, &regions);
      std::set<MachineBasicBlock*> visited;

      for (auto &r : regions) {
        auto &liveIns = gather->getIdemLiveIns(&r->getEntry());
        liveIns.erase(oldReg);
        liveIns.insert(usedReg);

        // for an iteration of each live-in register, renew the visited set.
        auto beginItr = ++MachineBasicBlock::iterator(r->getEntry());
        visited.clear();
        collectAntiDepsTrace(usedReg, beginItr, r->getEntry().getParent()->end(),
                             r->getEntry().getParent(),
                             std::vector<MIOp>(),
                             std::vector<MIOp>(), visited);

        // Update the live in registers for the region where insertedPos instr resides
        auto mbb = r->getEntry().getParent();
        visited.clear();
        bool seenRedef = false;
        updateLiveInOfPriorRegions(MachineBasicBlock::reverse_iterator(r->getEntry()),
                                   mbb->rend(), mbb, visited, oldReg, seenRedef);
      }
    }
  }
}

SmallVector<unsigned, 32> regUse;

void IdemRegisterRenamer::processHandledIntervals(std::vector<LiveIntervalIdem *> &handled,
                                                  unsigned currentStart) {
  for (auto interval : handled) {
    if (interval->endNumber() < currentStart) {
      if (TargetRegisterInfo::isPhysicalRegister(interval->reg)) {
        regUse[interval->reg] = 0;
        const unsigned *as = tri->getAliasSet(interval->reg);
        for (; as && *as; ++as)
          regUse[*as] = 0;
      }
    }
  }
}

bool IdemRegisterRenamer::numberOfSubLiveIntervalLessThanThreshold(LiveIntervalIdem *interval) {
  long res =  std::distance(interval->usepoint_begin(), interval->usepoint_end());
  return res < SpilledIntervalThreshold;
}

void IdemRegisterRenamer::assignRegOrStackSlotAtInterval(LiveIntervalIdem *interval,
                                                         std::vector<LiveIntervalIdem *> &handled,
                                                         std::vector<LiveIntervalIdem *> &spilled) {
  unsigned freeReg = 0;
  std::set<unsigned> allocables;
  getAllocableRegs(interval->reg, allocables);

  DenseSet<unsigned> unallocableRegs;
  addRegisterWithSubregs(unallocableRegs, interval->reg);
  addRegisterWithSuperRegs(unallocableRegs, interval->reg);
  for (unsigned r : regUse) {
    if (regUse[r]) {
      addRegisterWithSubregs(unallocableRegs, r);
      addRegisterWithSuperRegs(unallocableRegs, r);
    }
  }

  for (int r = reservedRegs.find_first(); r != -1; r = reservedRegs.find_next(r)) {
    addRegisterWithSuperRegs(unallocableRegs, r);
    addRegisterWithSubregs(unallocableRegs, r);
  }

  // remove those registers which will cause anti-dependence after renaming
  MachineInstr *mi = interval->usepoint_begin()->mo->getParent();
  countDefRegRaiseAntiDep(mi, unallocableRegs);

  for (auto r : unallocableRegs)
    if (allocables.count(r))
      allocables.erase(r);

  auto temp = allocables;
  set_intersect(temp, unallocableRegs);
  set_subtract(allocables, temp);

  if (!allocables.empty())
     freeReg = *allocables.begin();

  if (freeReg == 0) {
    // select a handled interval to be spilled out into memory.
    std::vector<LiveIntervalIdem*>::iterator spilledItr = handled.end();
    for (auto itr = handled.begin(), end = handled.end(); itr != end; ++itr) {
      if ((legalToReplace((*itr)->reg, interval->reg) &&
          !unallocableRegs.count((*itr)->reg) /*&&
          numberOfSubLiveIntervalLessThanThreshold(*itr)*/) &&
          (spilledItr == end || ((*itr)->costToSpill < (*spilledItr)->costToSpill)))
        spilledItr = itr;
    }

    assert(spilledItr != handled.end());
    bool res = getSpilledSubLiveInterval(*spilledItr, spilled);
    assert(res && "Should allocate register for those interval with infinity weight!");
    freeReg = (*spilledItr)->reg;
    handled.erase(spilledItr);
  }

  assert(freeReg != 0 && "No free register found!");
  regUse[freeReg] = 1;
  interval->reg = freeReg;
  li->insertOrCreateInterval(freeReg, interval);
}

void IdemRegisterRenamer::revisitSpilledInterval(std::vector<LiveIntervalIdem *> &spilled) {
  IntervalMap unhandled;
  std::vector<LiveIntervalIdem *> handled;
  regUse.resize(tri->getNumRegs(), 0);

  std::for_each(li->interval_begin(), li->interval_end(), [&](std::pair<unsigned, LiveIntervalIdem*> itr) {
    handled.push_back(itr.second);
    regUse[itr.second->reg] = 1;
  });

  std::for_each(spilled.begin(), spilled.end(), [&](LiveIntervalIdem *pIdem) {
    unhandled.push(pIdem);
  });

  getOrGroupId(spilled);

  while (!unhandled.empty()) {
    auto cur = unhandled.top();
    unhandled.pop();
    if (!cur->empty()) {
      processHandledIntervals(handled, cur->beginNumber());
    }

    // Allocating another register for current live interval.
    // Note that, only register is allowed to assigned to current interval.
    // Because the current interval corresponds to spilling code.
    std::vector<LiveIntervalIdem *> localSpilled;
    assignRegOrStackSlotAtInterval(cur, handled, localSpilled);
    getOrGroupId(localSpilled);
    std::for_each(localSpilled.begin(), localSpilled.end(), [&](LiveIntervalIdem *idem) {
      unhandled.push(idem);
    });

    // Insert spilling code for spilled live interval
    insertSpillingCodeForInterval(cur);
    handled.push_back(cur);
  }
}
#endif

void IdemRegisterRenamer::initializeIntervalSet(BitVector &allocSet) {
  active.clear();
  inactive.clear();
  handled.clear();
  interval2AssignedRegMap.clear();
  interval2StackSlotMap.clear();
  cur = nullptr;
  unhandled = IntervalMap();

  for (auto itr = li->interval_begin(), end = li->interval_end(); itr != end; ++itr) {
    assert(TargetRegisterInfo::isPhysicalRegister(itr->first));
    // we don't consider the unallocable registers.
    if (!allocSet[itr->first])
      continue;
    itr->second->reg = itr->first;
    active.push_back(itr->second);
  }
}

void IdemRegisterRenamer::prehandled(unsigned position) {
  // check for intervals in active that are expired or inactive.
  for (auto itr = active.begin(); itr != active.end(); ) {
    LiveIntervalIdem *interval = *itr;
    if (interval->isExpiredAt(position)) {
      itr = active.erase(itr);
      handled.push_back(interval);
    }
    else if (!interval->isLiveAt(position)) {
      itr = active.erase(itr);
      inactive.push_back(interval);
    }
    else
      ++itr;
  }

  // checks for intervals in inactive that are expired or active.
  for (auto itr = inactive.begin(); itr != inactive.end(); ) {
    LiveIntervalIdem *interval = *itr;
    if (interval->isExpiredAt(position)) {
      itr = inactive.erase(itr);
      handled.push_back(interval);
    }
    else if (interval->isLiveAt(position)) {
      itr = inactive.erase(itr);
      active.push_back(interval);
    }
    else
      ++itr;
  }
}

unsigned IdemRegisterRenamer::findOptimalSplitPos(LiveIntervalIdem *it,
                                                  unsigned minSplitPos,
                                                  unsigned maxSplitPos) {
  if (minSplitPos == maxSplitPos)
    return minSplitPos;

  MachineBasicBlock *minBlock = li->getBlockAtId(minSplitPos);
  MachineBasicBlock *maxBlock = li->getBlockAtId(maxSplitPos);
  if (minBlock == maxBlock)
    return maxSplitPos;

  if (it->hasHoleBetween(maxSplitPos - 1, maxSplitPos) &&
      !li->isBlockBegin(maxSplitPos)) {
    // Do not move split position if the interval has a hole before
    // maxSplitPos. Intervals resulting from Phi-Functions have
    // more than one definition with a hole before each definition.
    // When the register is needed for the second definition, an
    // earlier reloading is unnecessary.
    return maxSplitPos;
  }
  else
    return findOptimalSplitPos(minBlock, maxBlock, maxSplitPos);
}

unsigned IdemRegisterRenamer::findOptimalSplitPos(MachineBasicBlock *minBlock,
                                                  MachineBasicBlock *maxBlock,
                                                  unsigned maxSplitPos) {
  // Try to split at end of maxBlock. If this would be after
  // maxSplitPos, then use the begin of maxBlock
  unsigned optimalSplitPos = li->getIndex(&maxBlock->back()) + 2;
  if (optimalSplitPos > maxSplitPos)
    optimalSplitPos = li->getIndex(&maxBlock->front());

  int fromBlockId = minBlock->getNumber();
  int toBlockId = maxBlock->getNumber();
  unsigned minLoopDepth = ml->getLoopDepth(maxBlock);

  for (int i = toBlockId - 1; i >= fromBlockId; --i) {
    MachineBasicBlock *curMBB = mf->getBlockNumbered(i);
    unsigned depth = ml->getLoopDepth(curMBB);
    if (depth < minLoopDepth) {
      minLoopDepth = depth;
      optimalSplitPos = li->getIndex(&curMBB->back()) + 2;
    }
  }

  return optimalSplitPos;
}

LiveIntervalIdem* IdemRegisterRenamer::splitBeforeUsage(LiveIntervalIdem *it,
                                                        unsigned minSplitPos,
                                                        unsigned maxSplitPos) {
  assert(minSplitPos <= maxSplitPos);

  unsigned optimalSplitPos = findOptimalSplitPos(it, minSplitPos, maxSplitPos);
  assert(minSplitPos <= optimalSplitPos && optimalSplitPos <= maxSplitPos);
  if (optimalSplitPos == cur->endNumber())
    // If the optimal split position is at the end of current interval,
    // so splitting is not at all necessary.
    return nullptr;

  LiveIntervalIdem *rightPart = li->split(optimalSplitPos, it);
  rightPart->setInsertedMove();
  return rightPart;
}

void IdemRegisterRenamer::splitAndSpill(LiveIntervalIdem *it,
                                        unsigned startPos,
                                        unsigned endPos,
                                        bool isActive) {
  if (isActive) {
    unsigned minSplitPos = startPos + 1;
    unsigned maxSplitPos = std::min(it->getUsePointAfter(minSplitPos), it->endNumber());
    LiveIntervalIdem *splitChildren = splitBeforeUsage(it, minSplitPos, maxSplitPos);
    unhandled.push(splitChildren);
    splitForSpilling(it, startPos);

    // insert a store instruction after the last use point of it
    /*LiveIntervalIdem *parent = it->getSplitParent();
    const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(it->reg);
    int frameIndex;
    if (hasFrameSlot(parent))
      frameIndex = getFrameIndex(parent);
    else {
      frameIndex = mfi->CreateSpillStackObject(rc->getSize(), rc->getAlignment());
      setFrameIndex(parent, frameIndex);
    }
    if (!it->usePoints.empty()) {
      MachineInstr *mi = it->usePoints.rbegin()->mo->getParent();
      MachineInstr *st;
      if (mi == &mi->getParent()->back()) {
        tii->storeRegToStackSlot(*mi->getParent(), mi->getParent()->end(),
            it->reg, false, frameIndex, rc, tri);
      }
      else {
        MachineInstr *pos = getNextMI(mi);
        tii->storeRegToStackSlot(*mi->getParent(), pos,
                                 it->reg, false, frameIndex, rc, tri);
      }
      st = getNextMI(mi);
      li->mi2Idx[st] = li->getIndex(mi) + 1;

      if (splitChildren != nullptr) {
        minSplitPos = endPos + 1;
        maxSplitPos = splitChildren->getUsePointAfter(endPos);
        splitChildren = splitBeforeUsage(splitChildren, minSplitPos, maxSplitPos);
        if (splitChildren) {
          // insert a load instruction before the first use point of splitChildren
          if (!splitChildren->usePoints.empty()) {
            mi = splitChildren->usepoint_begin()->mo->getParent();
            tii->loadRegFromStackSlot(*mi->getParent(), mi, it->reg, frameIndex, rc, tri);
            auto ld = getPrevMI(mi);
            li->mi2Idx[ld] = li->getIndex(mi) - 1;
          }
        }
      }
    }*/
  }
  else {
    assert(it->hasHoleBetween(startPos - 1, startPos + 1));
    unhandled.push(splitBeforeUsage(it, startPos +1, startPos + 1));

    /*auto itr = it->upperBound(it->begin(), it->end(), endPos);
    if (itr == it->end())
      itr = RangeIterator(it->getLast());
    else
      --itr;

    assert(!itr->contains(endPos));*/
    // if itr doesn't contain the end position, which indicates there is a hole containing
    // [startPos, endPos)
  }
}

void IdemRegisterRenamer::splitForSpilling(LiveIntervalIdem *it,
                                           unsigned startPos) {
  unsigned maxSplitPos = startPos;
  unsigned minSplitPos = std::max(it->getUsePointBefore(maxSplitPos) + 1, it->beginNumber());

  // the whole interval is never used, so spill it entirely to memory
  if (minSplitPos == it->beginNumber()) {
    assert(it->getFirstUse() > startPos);
    assignInterval2StackSlot(it);

    LiveIntervalIdem *parent = it;
    while (parent && parent->isSplitChildren()) {
      parent = parent->getSplitChildBeforeOpId(parent->beginNumber());
      if (isAssignedPhyReg(parent)) {
        if (parent->getFirstUse() == UINT32_MAX)
          assignInterval2StackSlot(parent);
        else
          // exit
          parent = nullptr;
      }
    }
  }
  else {
    unsigned optimalSplitPos = findOptimalSplitPos(it, minSplitPos, maxSplitPos);
    LiveIntervalIdem *splitedChild = li->split(optimalSplitPos, it);
    assignInterval2StackSlot(splitedChild);
    insertMove(optimalSplitPos, it, splitedChild);
  }
}

void IdemRegisterRenamer::insertMove(unsigned insertedPos,
                                     LiveIntervalIdem *srcIt,
                                     LiveIntervalIdem *dstIt) {
  // output all moves here. When source and target are equal, the move is
  // optimized away later in assignRegNums
  insertedPos = (insertedPos + 1) & ~1;
  MachineInstr *mi = li->getMachineInstr(insertedPos);
  resolver->insertMoveInstr(mi);
  resolver->addMapping(srcIt, dstIt);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
unsigned IdemRegisterRenamer::allocateBlockedRegister(LiveIntervalIdem *interval,
                                                      BitVector &allocSet) {
  // TODO, need to refine, 11/14/2018
  std::map<unsigned, unsigned> freeUntilPos, blockPosBy;
  for (int reg = allocSet.find_first(); reg != -1 ; reg = allocSet.find_next(reg)) {
    freeUntilPos[reg] = UINT32_MAX;
    blockPosBy[reg] = UINT32_MAX;
  }

  for (LiveIntervalIdem *itr : active) {
    if (TargetRegisterInfo::isVirtualRegister(itr->reg))
      freeUntilPos[interval2AssignedRegMap[itr]] =
          interval->getUsePointAfter(interval->beginNumber());
    else
      blockPosBy[itr->reg] = 0;
  }

  for (LiveIntervalIdem *itr : inactive) {
    if (!itr->intersects(interval))
      continue;

    if (TargetRegisterInfo::isPhysicalRegister(itr->reg))
      blockPosBy[itr->reg] = itr->intersectAt(interval)->start;
    else
      freeUntilPos[itr->reg] = interval->getUsePointAfter(interval->beginNumber());
  }

  int reg = -1;
  std::for_each(freeUntilPos.begin(), freeUntilPos.end(), [&](const std::pair<unsigned, unsigned> &pair) {
    if (reg == -1 || pair.second > freeUntilPos[reg])
      reg = pair.first;
  });

  assert(reg != -1);
  int firstUseOfCur = interval->getFirstUse();
  if (freeUntilPos[reg] <= firstUseOfCur) {
    // all active and inactive interval are used before first use of current interval.
    // so we need to spill current interval and split it at an optimal position
    // before firstUseOfCur.
    LiveIntervalIdem *splitedChild = splitBeforeUsage(interval, freeUntilPos[reg] + 1, firstUseOfCur);
    unhandled.push(splitedChild);

    // Return 0 indicates we can't allocate the current interval with a register
    return 0;
  }

  unsigned splitPos = blockPosBy[reg];
  bool needSplit = splitPos <= cur->endNumber();
  assert(splitPos > 0);
  assert(needSplit || splitPos > cur->beginNumber());

  // register not available for full interval : so split it, assign the current interval to reg.
  interval2AssignedRegMap[interval] = reg;
  if (needSplit)
    unhandled.push(splitIntervalWhenPartialAvailable(interval, splitPos));

  // perform splitting and spilling for all affected intervals
  for (auto itr = active.begin(), end = active.end(); itr != end; ) {
    if ((*itr)->reg != reg)
      continue;

    if ((*itr)->intersects(interval)) {
      itr = active.erase(itr);
      splitAndSpill(*itr, interval->beginNumber(), interval->endNumber(), true);
    }
    else
      ++itr;
  }

  for (auto itr = inactive.begin(), end = inactive.end(); itr != end;) {
    if ((*itr)->reg != reg)
      continue;

    if ((*itr)->intersects(interval)) {
      itr = inactive.erase(itr);
      splitAndSpill(*itr, interval->beginNumber(), interval->endNumber(), false);
    }
  }
  return reg;
}
#pragma GCC diagnostic pop

unsigned IdemRegisterRenamer::getFreePhyReg(LiveIntervalIdem *interval,
                                            BitVector &allocSet) {
  if (allocSet.count() == 0)
    return 0;

  unsigned vreg = interval->reg;
  assert(TargetRegisterInfo::isVirtualRegister(vreg));
  std::map<unsigned, unsigned> freeUntilPos;

  for (int r = allocSet.find_first(); r != -1 ; r = allocSet.find_next(r))
    freeUntilPos[r] = UINT32_MAX;

  for (LiveIntervalIdem *itr : active) {
    unsigned reg = TargetRegisterInfo::isPhysicalRegister(itr->reg) ?
        itr->reg : interval2AssignedRegMap[itr];
    assert(allocSet[reg]);
    freeUntilPos[reg] = 0;
  }
  for (LiveIntervalIdem *itr : inactive) {
    if (itr->intersects(interval)) {
      unsigned reg = TargetRegisterInfo::isPhysicalRegister(itr->reg) ?
          itr->reg : interval2AssignedRegMap[itr];
      assert(allocSet[reg]);
      freeUntilPos[reg] = itr->intersectAt(interval)->start;
    }
  }

  int reg = -1;
  std::for_each(freeUntilPos.begin(), freeUntilPos.end(), [&](const std::pair<unsigned, unsigned> &pair){
    if (reg == -1 || pair.second > freeUntilPos[reg])
      reg = pair.first;
  });
  assert(reg != -1);

  if (freeUntilPos[reg] == 0) {
    // allocation failed
    return 0;
  }
  else if (freeUntilPos[reg] > interval->endNumber()) {
    // assign this reg to the current interval
    return reg;
  }
  else {
    // register available for first part of current interval.
    // split current at optimal position before freePos[reg].
    unhandled.push(splitIntervalWhenPartialAvailable(interval, freeUntilPos[reg]));
    return reg;
  }
}

LiveIntervalIdem *IdemRegisterRenamer::splitIntervalWhenPartialAvailable(LiveIntervalIdem *it,
                                                                         unsigned regAvaiUntil) {
  unsigned minSplitPos = std::max(it->getUsePointBefore(regAvaiUntil), it->beginNumber());
  return splitBeforeUsage(it, minSplitPos, regAvaiUntil);
}

int IdemRegisterRenamer::assignInterval2StackSlot(LiveIntervalIdem *interval) {
  assert(interval);
  unsigned vreg = interval->reg;
  assert(TargetRegisterInfo::isVirtualRegister(vreg));
  assert(!interval2StackSlotMap.count(interval));
  assert(!interval2AssignedRegMap.count(interval));
  auto rc = mri->getRegClass(vreg);
  int fi = mfi->CreateSpillStackObject(rc->getSize(), rc->getAlignment());
  interval2StackSlotMap[cur] = fi;
  return fi;
}

void IdemRegisterRenamer::linearScan(BitVector &allocSet) {
  while (!unhandled.empty()) {
    cur = unhandled.top();
    unhandled.pop();

    unsigned position = cur->beginNumber();
    // pre-handling, like move expired interval from active to handled list.
    prehandled(position);

    // if we find a free register, we are done: assign this virtual to
    // the free physical register and add this interval to the active
    // list.
    unsigned newReg;
    newReg = getFreePhyReg(cur, allocSet);
    if (newReg) {
      // Records the assigned phy register
      interval2AssignedRegMap[cur] = newReg;
      active.push_back(cur);
      continue;
    }
    newReg = allocateBlockedRegister(cur, allocSet);
    if (newReg) {
      interval2AssignedRegMap[cur] = newReg;
      active.push_back(cur);
    }
    else {
      // Otherwise, current interval would be splited and spill it's right part
      // into stack.
      assignInterval2StackSlot(cur);
      handled.push_back(cur);
    }
  }
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

void IdemRegisterRenamer::spillCurrentUse(AntiDeps &pair,
                                          MachineInstr *insertedPos,
                                          DenseSet<unsigned> &unallocableRegs) {
  assert(insertedPos && "Insertion position should not be null");
  auto useMO = pair.uses.front();
  const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(pair.reg);
  unsigned vreg = mri->createVirtualRegister(rc);
  MachineBasicBlock *mbb = insertedPos->getParent();

  // insert following instruction sequence right before insertedPos
  // 1. vreg = pair.reg
  // 2. str vreg, #FI
  emitRegToReg(*mbb, insertedPos, DebugLoc(), vreg, pair.reg, true);
  int slotIndex = mfi->CreateSpillStackObject(rc->getSize(), rc->getAlignment());
  tii->storeRegToStackSlot(*mbb, insertedPos, vreg, true, slotIndex, rc, tri);
  auto st = getPrevMI(insertedPos);
  unsigned id = li->getIndex(insertedPos);
  li->setIndex(id - 2, st);
  auto copyMI = getPrevMI(st);
  li->setIndex(id - 4, copyMI);

  // insert a load instruction right before the use first
  tii->loadRegFromStackSlot(*mbb, useMO.mi, vreg, slotIndex, rc, tri);
  auto ld = getPrevMI(useMO.mi);
  unsigned ldId = li->getIndex(useMO.mi) - 2;
  li->setIndex(ldId, ld);
  // replace all references to pair.use with the vreg
  std::for_each(pair.uses.begin(), pair.uses.end(), [&](MIOp &op) {
    op.mi->getOperand(op.index).setReg(vreg);
  });


  // remove some interval part for pair.reg
  LiveIntervalIdem *interval = li->intervals[pair.reg];
  assert(interval);
  interval->removeRange(id-1, li->getIndex(pair.uses.back().mi) + 1);

  // insert a new interval for vreg, perform register allocation
  LiveIntervalIdem *newInterval = new LiveIntervalIdem;
  newInterval->oldReg = pair.reg;
  newInterval->reg = vreg;

  newInterval->addRange(id-2, li->getIndex(pair.uses.back().mi) + 1);
  newInterval->addUsePoint(id-2, &copyMI->getOperand(0));
  newInterval->addUsePoint(id-1, &st->getOperand(0));
  newInterval->addUsePoint(ldId, &ld->getOperand(0));

  std::for_each(pair.uses.begin(), pair.uses.end(), [&](MIOp &op) {
    newInterval->addUsePoint(li->getIndex(op.mi), &op.mi->getOperand(op.index));
  });

  li->computeCostToSpill(newInterval);
  choosePhysRegForRenaming(newInterval, unallocableRegs);
}

void IdemRegisterRenamer::choosePhysRegForRenaming(LiveIntervalIdem *interval,
                                                   DenseSet<unsigned> &unallocableRegs) {
  auto allocSet = tri->getAllocatableSet(*mf);
  auto numRegs = allocSet.size();

  unsigned oldReg = interval->oldReg;
  assert(oldReg && TargetRegisterInfo::isPhysicalRegister(oldReg));
  /*if (mf->getFunction()->getName() == "flipbit") {
    llvm::dbgs() << "unallocables: \n";
    std::for_each(unallocableRegs.begin(), unallocableRegs.end(), [&](unsigned r) {
      llvm::dbgs() << tri->getName(r) << ",";
    });
    llvm::dbgs() << "\n";
  }*/

  // Remove some registers are not available when making decision of choosing.
  for (unsigned i = 0; i < numRegs; i++) {
    if (allocSet[i] && (unallocableRegs.count(i) || !legalToReplace(i, oldReg)))
      allocSet.reset(i);
  }

  /*if (mf->getFunction()->getName() == "flipbit") {
    llvm::dbgs() << "allocables after filter:\n";
    for (unsigned i = 0; i < numRegs; i++)
      if (allocSet[i])
        llvm::dbgs() << tri->getName(i) << ",";
    llvm::dbgs() << "\n";
  }*/

  // obtains a free register used for move instr.
  // choose an interval to be evicted into memory, and insert spilling code as
  // appropriate.
  initializeIntervalSet(allocSet);
  resolver = new MoveResolver(tri, tii, numRegs, this);
  unhandled.push(interval);
  linearScan(allocSet);

  if (!active.empty())
    handled.insert(handled.end(), active.begin(), active.end());
  if (!inactive.empty())
    handled.insert(handled.end(), inactive.begin(), inactive.end());

  // Resolve move
  resolveDataflow();

  if (!rewriter)
    rewriter = new VirRegRewriter;

  rewriter->rewrite(handled, interval2AssignedRegMap);
  delete resolver;
}

void IdemRegisterRenamer::resolveDataflow() {
  std::set<MachineBasicBlock*> uniqueMBBs;

  for (auto &mbb : *mf) {
    if (!mbb.succ_empty()) {
      for (auto succ = mbb.succ_begin(), end = mbb.succ_end(); succ != end; ++succ) {
        MachineBasicBlock *succBB = *succ;
        if (!uniqueMBBs.insert(succBB).second)
          continue;

        std::set<unsigned> liveins = li->liveIns[succBB->getNumber()];
        if (liveins.empty())
          continue;
        if (mbb.empty() || succBB->empty())
          continue;

        for (unsigned reg : liveins) {
          LiveIntervalIdem *parent = li->intervals[reg];
          assert(parent);
          parent = parent->getSplitParent();

          LiveIntervalIdem *srcIt = parent->getSplitChildAtOpId(li->getIndex(&mbb.back()));
          LiveIntervalIdem *dstIt = parent->getSplitChildAtOpId(li->getIndex(&succBB->front()));
          if (srcIt && dstIt && srcIt != dstIt)
            resolver->addMapping(srcIt, dstIt);
        }

        // find a position to insert a move instruction.
        findPosAndInsertMove(&mbb, succBB);
        // resolve map
        resolver->resolveMapping();
      }
      uniqueMBBs.clear();
    }
  }
}

void IdemRegisterRenamer::findPosAndInsertMove(MachineBasicBlock *src, MachineBasicBlock *dst) {
  if (std::distance(src->succ_begin(), src->succ_end()) <= 1)
    // insert a move instruction at the end of source basic block.
    resolver->insertMoveInstr(&src->back());
  else
    // insert a move instruction at the beginning of destination block.
    resolver->insertMoveInstr(&dst->front());
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
                                                 std::vector<MIOp> uses,
                                                 std::vector<MIOp> defs,
                                                 bool &canReplace,
                                                 bool seeIdem) {
  if (!mbb)
    return;
  if (!visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    auto mi = begin;
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
          uses.emplace_back(mi, i);
      }
      else {
        // mo.isDef()
        if (!uses.empty()) {
          canReplace = false;
          return;
        }
        defs.emplace_back(mi, i);
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
                             uses,
                             defs,
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
  std::vector<MIOp> uses, defs;
  walkDFSToGatheringUses(def->getReg(),
      // Skip current mi defines the def operand, starts walk through from next mi.
      ++MachineBasicBlock::iterator(def->getParent()),
      mbb->end(), mbb, visited, uses, defs, canReplace, false);

  usesAndDefs.insert(usesAndDefs.end(), uses.begin(), uses.end());
  usesAndDefs.insert(usesAndDefs.end(), defs.begin(), defs.end());
}

void IdemRegisterRenamer::collectUnallocableRegs(AntiDeps &pair,
                                                 MachineInstr *&insertedPos,
                                                 std::vector<IdempotentRegion *> &regions,
                                                 DenseSet<unsigned> &unallocableRegs) {
  std::set<MachineBasicBlock*> visited;
  unsigned minIndex = UINT32_MAX;
  auto useMO = pair.uses.front();

  for (auto r : regions) {
    MachineInstr &idem = r->getEntry();
    auto liveins = gather->getIdemLiveIns(&idem);
    std::for_each(liveins.begin(), liveins.end(), [&](unsigned reg) {
      addRegisterWithSubregs(unallocableRegs, reg);
    });

    auto begin = MachineBasicBlock::reverse_iterator(idem);
    visited.clear();
    collectUnallocableRegsDFS(begin, idem.getParent()->rend(),
                              idem.getParent(), visited, unallocableRegs);

    unsigned index = li->getIndex(&idem);
    if (index < minIndex) {
      minIndex = index;
      insertedPos = &idem;
    }
  }

  // can not assign the old register to use mi
  addRegisterWithSubregs(unallocableRegs, pair.reg);
  auto begin = pair.uses.begin(), end = pair.uses.end();
  std::for_each(begin, end, [&](MIOp &op) {
    MachineInstr *useMI = op.mi;
    for (unsigned j = 0, e = useMI->getNumOperands(); j < e; j++) {
      auto mo = useMI->getOperand(j);
      if (mo.isReg() && mo.getReg() && mo.isDef())
        addRegisterWithSubregs(unallocableRegs, mo.getReg());
    }
  });

  countRegistersRaiseAntiDep(useMO.mi, useMO.mi->getParent()->end(), useMO.mi->getParent(), unallocableRegs);
}

void IdemRegisterRenamer::collectUnallocableRegsDFS(MachineBasicBlock::reverse_iterator begin,
                                                    MachineBasicBlock::reverse_iterator end,
                                                    MachineBasicBlock *mbb,
                                                    std::set<MachineBasicBlock *> &visited,
                                                    DenseSet<unsigned> &unallocableRegs) {
  if (!mbb || !visited.insert(mbb).second)
    return;

  for (; begin != end; ++begin) {
    if (tii->isIdemBoundary(&*begin)) {
      auto liveIns = gather->getIdemLiveIns(&*begin);
      std::for_each(liveIns.begin(), liveIns.end(), [&](unsigned reg){
        addRegisterWithSubregs(unallocableRegs, reg);
        addRegisterWithSuperRegs(unallocableRegs, reg);
      });
      return;
    }
  }

  std::for_each(mbb->pred_begin(), mbb->pred_end(), [&](MachineBasicBlock *pred){
    collectUnallocableRegsDFS(pred->rbegin(), pred->rend(), pred, visited, unallocableRegs);
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

bool IdemRegisterRenamer::shouldSpillCurrent(unsigned reg,
                                             DenseSet<unsigned> &unallocableRegs,
                                             std::vector<IdempotentRegion *> &regions) {
  unsigned numStimulateousLiveInReg = 0;
  unsigned freeRegs = getNumFreeRegs(reg, unallocableRegs);

  std::for_each(regions.begin(), regions.end(), [&](IdempotentRegion* r) {
    numStimulateousLiveInReg += region2NumAntiDeps[&r->getEntry()];
  });

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
      if (!mo.isReg() || !mo.getReg() || !partialEquals(mo.getReg(), reg) || mo.isUse())
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

void IdemRegisterRenamer::countRegistersRaiseAntiDep(MachineBasicBlock::iterator begin,
                                                     MachineBasicBlock::iterator end,
                                                     MachineBasicBlock *mbb,
                                                     DenseSet<unsigned> &unallocableRegs) {
  if (!mbb)
    return;

  for (; begin != end; ++begin) {
    if (tii->isIdemBoundary(&*begin))
      return;
    for (unsigned i = 0, e = begin->getNumOperands(); i < e; i++) {
      auto mo = begin->getOperand(i);
      if (mo.isReg() && mo.isDef() &&
          TargetRegisterInfo::isPhysicalRegister(mo.getReg()) &&
          !reservedRegs[mo.getReg()]) {
        addRegisterWithSubregs(unallocableRegs, mo.getReg());
        addRegisterWithSuperRegs(unallocableRegs, mo.getReg());
      }
    }
  }

  for (auto itr = mbb->succ_begin(),succEnd = mbb->succ_end(); itr != succEnd; ++itr) {
    auto succ = *itr;
    if (!dt->dominates(succ, mbb))
      countRegistersRaiseAntiDep(succ->begin(), succ->end(), succ, unallocableRegs);
    else
      countRegisterRaiseAntiDepsInLoop(succ->begin(), succ->end(), succ, unallocableRegs);
  }
}

static int m;

bool IdemRegisterRenamer::handleAntiDependences() {
  if (antiDeps.empty())
    return false;

  std::vector<IdempotentRegion *> regions;
  while (!antiDeps.empty()) {
    // llvm::dbgs()<<antiDeps.size()<<"\n";
    auto pair = antiDeps.front();
    antiDeps.pop_front();

    ++m;
    // llvm::dbgs()<<m++<<"\n";
    // mf->dump();

    if (pair.uses.empty() || pair.defs.empty())
      continue;

    auto &useMO = pair.uses.front();
    auto &defMO = pair.defs.back();
    mir->getRegionsContaining(*useMO.mi, &regions);
    if (regions.empty())
      continue;

    bool useOverlapped = useMO.mi->getOperand(useMO.index).getReg() != pair.reg;
    bool defOverlapped = !partialEquals(defMO.mi->getOperand(defMO.index).getReg(), pair.reg);
    // FIXME, 11/15/2018, try to reduce redundant inserted move instruction along different paths
    if (useOverlapped) {
      //======================================================================================= //
      // Handle the case that uses of different anti-dependence overlap
      //======================================================================================= //
      std::vector<MIOp>::iterator pos2 = pair.uses.end() - 1;
      if (isTwoAddressInstr(pos2->mi, pair.reg)) {
        MachineBasicBlock::iterator begin = pos2->mi->getParent()->begin(),
            end = pos2->mi->getParent()->end(), prev;
        for (; begin != end; ++begin) {
          if (begin == pos2->mi) {
            break;
          }
          prev = begin;
        }
        assert(prev != pos2->mi->getParent()->end());
        if (prev != MachineBasicBlock::iterator() && tii->isMovInstr(prev)) {
          if (prev->getOperand(0).getReg() == pair.reg) {
            // Such as,
            // R0 = R3
            // R0 = LDR_PRE R0, R1
            pair.defs.clear();
            pair.defs.emplace_back(prev, 0);
            pair.uses.pop_back();

            assert(prev->getOperand(1).isReg());
            unsigned destReg = prev->getOperand(1).getReg();

            pos2 = pair.uses.end() - 1;

            // Insert a move from old register to destination register.
            MachineBasicBlock::iterator itr = pos2->mi->getParent()->end(), insertedPos;
            MachineBasicBlock::iterator beginPos = pos2->mi;

            for (; itr != beginPos; --itr) {
              insertedPos = itr;
            }
            tii->copyPhysReg(*pos2->mi->getParent(), insertedPos, DebugLoc(), destReg, pair.reg, true);
            auto copyMI = getPrevMI(insertedPos);
            li->setIndex(li->getIndex(pos2->mi) + 2, copyMI);
            pair.uses.emplace_back(copyMI, 1);
          }
          else if (prev->getOperand(1).getReg() == pair.reg) {
            // Such as,
            // R3 = R0
            // R3 = LDR_PRE R3, R1
            continue;
          }
        }
      }
      auto pos = pair.uses.begin();
      while (pos != pair.uses.end()) {
        MIOp &op = *pos;
        if (op.mi->getOperand(op.index).getReg() != pair.reg) {
          // move the pos to the first modified MIOp.
          ++pos;
        } else
          break;
      }

      if (pos == pair.uses.end())
        // Reach here, it indicates all uses have been replaced. It isn't needed to advance.
        continue;

      if (pos != pair.uses.begin()) {
        // modified [begin, pos)
        unsigned useReg = pair.uses.front().mi->getOperand(pair.uses.front().index).getReg();
        DenseSet<unsigned> unallocableRegs;
        addRegisterWithSubregs(unallocableRegs, pair.reg);
        auto begin = pair.uses.begin();
        std::for_each(begin, pos, [&](MIOp &op) {
          MachineInstr *useMI = op.mi;
          for (unsigned j = 0, e = useMI->getNumOperands(); j < e; j++) {
            auto mo = useMI->getOperand(j);
            if (mo.isReg() && mo.getReg() && mo.isDef())
              addRegisterWithSubregs(unallocableRegs, mo.getReg());
          }
        });

        if (!unallocableRegs.count(useReg)) {
          unsigned from = li->getIndex(pair.uses.front().mi);
          unsigned to = li->getIndex((pos - 1)->mi);
          if (from < to)
            std::swap(from, to);

          LiveIntervalIdem *itrv = new LiveIntervalIdem;
          if (from > to)
            std::swap(from, to);

          itrv->addRange(from, to);
          std::for_each(begin, pos, [&](MIOp &op) {
            itrv->addUsePoint(li->getIndex(op.mi), &op.mi->getOperand(op.index));
          });

          bool interfere = false;
          for (auto itr = li->interval_begin(), end = li->interval_end(); itr != end; ++itr) {
            if (itr->second->intersects(itrv)) {
              interfere = true;
              break;
            }
          }

          if (!interfere) {
            std::for_each(begin, pos, [&](MIOp &op) {
              MachineInstr *useMI = op.mi;
              for (unsigned j = 0, e = useMI->getNumOperands(); j < e; j++) {
                auto mo = useMI->getOperand(j);
                if (mo.isReg() && mo.getReg() && mo.getReg() == pair.reg)
                  mo.setReg(useReg);
              }
            });

            itrv->reg = pair.reg;
            li->removeInterval(itrv);

            itrv->reg = useReg;
            li->insertOrCreateInterval(useReg, itrv);
            continue;
          }
        }

        // delete those modified uses.
        pair.uses.erase(pair.uses.begin(), pos);
        // if all uses have been replaced, just continue to cope with next
        if (pair.uses.empty())
          continue;
      }

      pos = pair.defs.begin();
      while (pos != pair.defs.end()) {
        if (pos->mi->getOperand(pos->index).getReg() != pair.reg)
          ++pos;
        else
          break;
      }

      // [begin, pos) have been modified.
      pair.defs.erase(pair.defs.begin(), pos);

      if (pair.defs.empty())
        continue;
    }
    if (defOverlapped) {
      //======================================================================================= //
      // Handle the case that defs of different anti-dependence overlap
      // We don't need to cope with those def registers have replaced in the previous round if exist
      //======================================================================================= //

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
      std::vector<MIOp>::iterator pos2 = pair.uses.end() - 1;
      if (isTwoAddressInstr(pos2->mi, pair.reg)) {
        MachineBasicBlock::iterator begin = pos2->mi->getParent()->begin(),
            end = pos2->mi->getParent()->end(), prev;
        for (; begin != end; ++begin) {
          if (begin == pos2->mi) {
            break;
          }
          prev = begin;
        }
        assert(prev != pos2->mi->getParent()->end());
        if (prev != MachineBasicBlock::iterator() && tii->isMovInstr(prev)) {
          if (prev->getOperand(0).getReg() == pair.reg) {
            // Such as,
            // R0 = R3
            // R0 = LDR_PRE R0, R1
            pair.defs.clear();
            pair.defs.emplace_back(prev, 0);
            pair.uses.pop_back();

            assert(prev->getOperand(1).isReg());
            unsigned destReg = prev->getOperand(1).getReg();

            pos2 = pair.uses.end() - 1;

            // Insert a move from old register to destination register.
            MachineBasicBlock::iterator itr = pos2->mi->getParent()->end(), insertedPos;
            MachineBasicBlock::iterator beginPos = pos2->mi;

            for (; itr != beginPos; --itr) {
              insertedPos = itr;
            }
            tii->copyPhysReg(*pos2->mi->getParent(), insertedPos, DebugLoc(), destReg, pair.reg, true);
            auto copyMI = getPrevMI(insertedPos);
            li->setIndex(li->getIndex(pos2->mi) + 2, copyMI);
            pair.uses.emplace_back(copyMI, 1);
          }
          else if (prev->getOperand(1).getReg() == pair.reg) {
            // Such as,
            // R3 = R0
            // R3 = LDR_PRE R3, R1
            continue;
          }
        }
      }

      auto pos = pair.defs.begin();
      while (pos != pair.defs.end()) {
        if (pos->mi->getOperand(pos->index).getReg() == pair.reg)
          ++pos;
        else
          break;
      }

      // delete those elements within the range [pos, end)
      pair.defs.erase(pos, pair.defs.end());
      if (pair.defs.empty())
        continue;

      // We should insert a extra move instruction right after the last un-replaced operand
      // pointed by pos.
      pos2 = pair.uses.begin();
      while (pos2 != pair.uses.end()) {
        if (pos2->mi->getOperand(pos2->index).getReg() == pair.reg)
          ++pos2;
        else
          break;
      }

      if (pos2 != pair.uses.end()) {
        // move the pos2 to the position where pos2 pointer to the last unmodified MIOp
        --pos2;
        unsigned destReg = pair.reg;
        unsigned srcReg = pair.reg;
        MachineBasicBlock::iterator itr = pos2->mi->getParent()->end(), insertedPos;
        MachineBasicBlock::iterator begin = pos2->mi;

        for (; itr != begin; --itr) {
          insertedPos = itr;
        }

        tii->copyPhysReg(*pos->mi->getParent(), insertedPos, DebugLoc(), destReg, srcReg, true);
        auto mov = getPrevMI(insertedPos);
        unsigned index = li->getIndex(itr)-2;
        li->setIndex(index, mov);

        ++pos2;
        pair.uses.erase(pos, pair.uses.end());
        pair.uses.emplace_back(mov, 1);
        // twoAddrInstExits = false;
      }
    }

    MachineInstr *intervalEnd = pair.uses.back().mi;
    MIOp &miLastDef = pair.defs.back();

    // Try to replace the old register name with other register to reduce
    // inserted move instruction.
    // If we can not find such register, than alter to insert move.

    // We just count on such situation that all uses are within the same region
    // as the current region.
    std::vector<MIOp> usesAndDef;
    bool canReplace = false;

    // get the last insertion position of previous adjacent region
    // or the position of prior instruction depends on if the current instr
    // is a two address instr.
    bool twoAddrInstExits = isTwoAddressInstr(pair.uses.back().mi, pair.reg);

    getUsesSetOfDef(&miLastDef.mi->getOperand(miLastDef.index), usesAndDef, canReplace);

    auto saved = miLastDef.mi;
    std::set<MachineBasicBlock *> visited;
    canReplace &= !twoAddrInstExits;
    if (canReplace)
      willRenameCauseOtherAntiDep(++saved, miLastDef.mi->getParent()->end(),
                                  miLastDef.mi->getParent(), pair.reg, visited, canReplace);

    // If the current reg is used in ret instr, we can't replace it.
    if (canReplace) {
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
        std::vector<MachineBasicBlock *> worklist;
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
      LiveIntervalIdem *itrvl = new LiveIntervalIdem;

      // indicates this interval should not be spilled out into memory.
      itrvl->costToSpill = UINT32_MAX;
      std::vector<MIOp>::iterator begin = pair.defs.begin(), end = pair.defs.end();

      if (from > to) {
        // this situation could occurs caused by loop.
        std::swap(from, to);
      }

      itrvl->addRange(from, to);
      for (auto itr = begin; itr != end; ++itr) {
        MachineInstr *mi = itr->mi;
        itrvl->usePoints.insert(UsePoint(li->getIndex(mi), &mi->getOperand(itr->index)));
      }

      for (auto &op : usesAndDef) {
        MachineInstr *mi = op.mi;
        itrvl->usePoints.insert(UsePoint(li->getIndex(mi), &mi->getOperand(op.index)));
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
        std::for_each(mbb->livein_begin(), mbb->livein_end(), [&](unsigned reg) {
          addRegisterWithSubregs(unallocableRegs, reg);
        });
      }

      for (auto itr = begin; itr != end; ++itr) {
        auto mbb = itr->mi->getParent();
        std::for_each(mbb->livein_begin(), mbb->livein_end(), [&](unsigned reg) {
          addRegisterWithSubregs(unallocableRegs, reg);
        });

        std::for_each(mbb->pred_begin(), mbb->pred_end(), [&](MachineBasicBlock *pred) {
          std::for_each(pred->livein_begin(), pred->livein_end(), [&](unsigned reg) {
            addRegisterWithSubregs(unallocableRegs, reg);
          });
        });
      }

      phyReg = getFreeRegisterForRenaming(pair.reg, itrvl, unallocableRegs);

      if (phyReg != 0) {
        // We have found a free register can be used for replacing the clobber register.
        for (auto itr = begin; itr != end; ++itr) {
          itr->mi->getOperand(itr->index).setReg(phyReg);
        }
        std::for_each(usesAndDef.begin(), usesAndDef.end(), [=](MIOp &mo) {
          mo.mi->getOperand(mo.index).setReg(phyReg);
        });

        // remove the live interval from old interval
        li->intervals[pair.reg]->removeRange(from, to);

        // Update the internal data structure of live interval analysis
        li->insertOrCreateInterval(phyReg, itrvl);

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
      else
        delete itrvl;
    }

    if (twoAddrInstExits) {
      // transform following code
      // IDEM
      // ...
      // R0 = LDr R1, R0
      // ... = R0
      // ...
      //
      // to
      // IDEM
      // ...
      // %vreg1 = %R0
      // %vreg1 = LDr R1, %vreg1
      // ... = %vreg1
      LiveIntervalIdem *oldInterval = li->intervals[pair.reg];
      assert(oldInterval);

      const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(pair.reg);
      unsigned vreg = mri->createVirtualRegister(rc);
      MachineInstr* lastUseMI = pair.uses.back().mi;
      MachineBasicBlock *mbb = lastUseMI->getParent();
      emitRegToReg(*mbb, lastUseMI, lastUseMI->getDebugLoc(), vreg, pair.reg, true);
      auto copyMI = getPrevMI(lastUseMI);
      unsigned id = li->getIndex(lastUseMI) - 2;
      unsigned from = id + 2;
      li->setIndex(id, copyMI);

      auto insertedPos = getNextMI(lastUseMI);
      emitRegToReg(*mbb, insertedPos, lastUseMI->getDebugLoc(), pair.reg, vreg, true);
      auto copyMI2 = getPrevMI(insertedPos);
      unsigned id2 = li->getIndex(lastUseMI) + 2;
      unsigned to = id2;
      li->setIndex(id, copyMI2);

      if (from > to)
        std::swap(from, to);

      oldInterval->removeRange(from, to);

      // create a new interval for vreg.
      LiveIntervalIdem *newInterval = new LiveIntervalIdem;
      newInterval->oldReg = pair.reg;
      newInterval->reg = vreg;
      newInterval->addRange(from, to);

      newInterval->addUsePoint(id, &copyMI->getOperand(0));
      newInterval->addUsePoint(id2, &copyMI2->getOperand(1));

      // add use points, and replace old reg with virtual register.
      for (unsigned i = 0, e = lastUseMI->getNumOperands(); i < e; i++) {
        MachineOperand &op = lastUseMI->getOperand(i);
        if (op.isReg() && op.getReg() == pair.reg) {
          op.setReg(vreg);
          newInterval->addUsePoint(id+2, &op);
        }
      }

      DenseSet<unsigned> unallocableRegs;
      for (auto r : regions) {
        MachineInstr &idem = r->getEntry();
        auto liveins = gather->getIdemLiveIns(&idem);
        std::for_each(liveins.begin(), liveins.end(), [&](unsigned reg) {
          addRegisterWithSubregs(unallocableRegs, reg);
          addRegisterWithSuperRegs(unallocableRegs, reg);
        });

        auto begin = MachineBasicBlock::reverse_iterator(idem);
        visited.clear();
        collectUnallocableRegsDFS(begin, idem.getParent()->rend(),
                                  idem.getParent(), visited, unallocableRegs);
      }

      // can not assign the old register to use mi
      addRegisterWithSubregs(unallocableRegs, pair.reg);
      addRegisterWithSuperRegs(unallocableRegs, pair.reg);

      countRegistersRaiseAntiDep(copyMI2, mbb->end(),mbb, unallocableRegs);

      // perform register allocation
      choosePhysRegForRenaming(newInterval, unallocableRegs);

      // transform the pair's uses list and defs list
      pair.uses.back() = MIOp(copyMI, 1);
      pair.defs.front() = MIOp(copyMI2, 0);
    }

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
    DenseSet<unsigned> unallocableRegs;
    collectUnallocableRegs(pair, insertedPos, regions, unallocableRegs);
    assert(insertedPos);

    if (shouldSpillCurrent(pair.reg, unallocableRegs, regions)) {
      spillCurrentUse(pair, insertedPos, unallocableRegs);
      continue;
    }

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
    const TargetRegisterClass *rc = tri->getMinimalPhysRegClass(pair.reg);
    unsigned vreg = mri->createVirtualRegister(rc);

    emitRegToReg(*insertedPos->getParent(), insertedPos, DebugLoc(), vreg, pair.reg, true);
    auto copy = getPrevMI(insertedPos);
    unsigned insertedPosId = li->getIndex(insertedPos);
    li->setIndex(insertedPosId - 2, copy);

    LiveIntervalIdem *interval = new LiveIntervalIdem;
    interval->oldReg = pair.reg;
    interval->reg = vreg;
    auto from = insertedPosId - 2;
    auto to = li->getIndex(intervalEnd);

    if (from > to) {
      // this situation could occurs by loop.
      std::swap(from, to);
    }

    interval->addRange(from, to);    // add an interval for a temporal move instr.
    // Add the destination reg of inserted copy mi as a use point.
    interval->addUsePoint(insertedPosId - 2, &copy->getOperand(0));
    // Add all uses in pair.uses as use points.
    size_t e = pair.uses.size() - twoAddrInstExits;
    for (size_t k = 0; k < e; k++) {
      MachineOperand &mo = pair.uses[k].mi->getOperand(pair.uses[k].index);
      assert(mo.getParent());
      mo.setReg(vreg);
      interval->addUsePoint(li->getIndex(mo.getParent()), &mo);
    }

    li->computeCostToSpill(interval);
    choosePhysRegForRenaming(interval, unallocableRegs);
  }
  return true;
}

void IdemRegisterRenamer::emitRegToReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI, 
                                       DebugLoc DL,
                                       unsigned DestReg, unsigned SrcReg,
                                       bool KillSrc) {
  auto mib = BuildMI(MBB, MI, DL, tii->get(TargetOpcode::COPY))
      .addReg(DestReg).addReg(SrcReg, KillSrc? RegState::Kill : 0);
  pseduoMoves.push_back(mib);
}

void IdemRegisterRenamer::eliminatePseudoMoves() {
  for (auto &mi : pseduoMoves) {
    tii->copyPhysReg(*mi->getParent(), mi, DebugLoc(),
                     mi->getOperand(0).getReg(),
                     mi->getOperand(1).getReg(),
                     mi->getOperand(1).isKill());
    mi->eraseFromParent();
  }
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
  ml = getAnalysisIfAvailable<MachineLoopInfo>();

  tii = MF.getTarget().getInstrInfo();
  tri = MF.getTarget().getRegisterInfo();
  mf = &MF;
  mri = &MF.getRegInfo();
  mfi = MF.getFrameInfo();
  reservedRegs = tri->getReservedRegs(*mf);
  m = 0;

  // Collects anti-dependences operand pair.
  /*llvm::errs() << "Before renaming2: \n";
  MF.dump();*/

  bool changed = false;
  collectLiveInRegistersForRegions();
  computeAntiDependenceSet();
  changed |= handleAntiDependences();

  // eliminatePseudoMoves();
  /*llvm::errs() << "After renaming2: \n";
  MF.dump();*/
  clear();
  return changed;
}
