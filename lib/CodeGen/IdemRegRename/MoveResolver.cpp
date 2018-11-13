//===----- MoveResolver.cpp - Move resolver for register renaming ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MoveResolver.h"

using namespace llvm;
void MoveResolver::insertMoveInstr(MachineInstr *pos)  {
  assert(pos);
  MachineBasicBlock *mbb = pos->getParent();
  if (insertedMBB && (insertedMBB != mbb || pos != insertedPos))
    // insertion position has been changed, resolve mappings.
    resolveMapping();

  insertedMBB = mbb;
  insertedPos = pos;
}

void MoveResolver::addMapping(LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt) {
  assert(srcIt && dstIt);
  if (srcIt == dstIt)
    return;

  mappings.emplace_back(srcIt, dstIt);
}

void MoveResolver::resolveMapping() {
  // Block all registers that are used as input operands of a move.
  // When a register is blocked, no move to this register is emitted.
  // This is necessary for detecting cycles in moves.
  for (auto i = mappings.size(); i >= 1; --i) {
    LiveIntervalIdem *it = mappings[i-1].first;
    if (it) blockRegisters(it);
  }

  size_t spillCandidate = mappings.size();
  while (!mappings.empty()) {
    bool processedInterval = false;
    for (auto i = mappings.size(); i >= 1; --i) {
      LiveIntervalIdem *srcIt = mappings[i-1].first;
      LiveIntervalIdem *dstIt = mappings[i-1].second;
      assert(srcIt && dstIt);
      if (isSafeToProcessMove(srcIt, dstIt)) {
        insertMove(srcIt, dstIt);
        unblockRegisters(srcIt);
        mappings.erase(mappings.begin()+i-1);
        processedInterval = true;
      }
      else if (renamer->isAssignedPhyReg(srcIt)) {
        // this interval cannot be processed now because target is not free
        // it starts in a register, so it is a possible candidate for spilling
        spillCandidate = i;
      }
    }

    if (!processedInterval) {
      // no move could be processed because there is a cycle in the move list
      // (e.g. r1 . r2, r2 . r1), so one interval must be spilled to memory
      LiveIntervalIdem *srcIt = mappings[spillCandidate].first;

      LiveIntervalIdem *spillInterval = new LiveIntervalIdem;
      // add pseduo-range
      // add a dummy range because real position is difficult to calculate
      // Note: this range is a special case when the integrity of the allocation is checked
      spillInterval->addRange(1, 2);

      renamer->assignInterval2StackSlot(spillInterval);
      insertMove(srcIt, spillInterval);
      mappings[spillCandidate].first = spillInterval;
      unblockRegisters(srcIt);
    }
  }
}

void MoveResolver::insertMove(LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt) {
  assert(srcIt && dstIt);
  if (srcIt == dstIt)
    return;

  if (renamer->isAssignedPhyReg(srcIt) && renamer->isAssignedPhyReg(dstIt)) {
    unsigned srcReg = renamer->getAssignedPhyReg(srcIt), dstReg = renamer->getAssignedPhyReg(dstIt);
    tii->copyPhysReg(*insertedMBB, insertedPos, insertedPos->getDebugLoc(), dstReg, srcReg, true);
  }
  else if (renamer->isAssignedStackSlot(srcIt) && renamer->isAssignedPhyReg(dstIt)) {
    int srcSlot = renamer->getAssignedStackSlot(srcIt);
    unsigned dstReg = renamer->getAssignedPhyReg(dstIt);
    const TargetRegisterClass *dstRC = tri->getMinimalPhysRegClass(dstReg);
    tii->loadRegFromStackSlot(*insertedMBB, insertedPos, dstReg, srcSlot, dstRC, tri);
  }
  else if (renamer->isAssignedPhyReg(srcIt) && renamer->isAssignedStackSlot(dstIt)) {
    int dstSlot = renamer->getAssignedStackSlot(dstIt);
    unsigned srcReg = renamer->getAssignedPhyReg(srcIt);
    const TargetRegisterClass *srcRC = tri->getMinimalPhysRegClass(srcReg);
    tii->storeRegToStackSlot(*insertedMBB, insertedPos, srcReg, true, dstSlot, srcRC, tri);
  }
  else {
    assert(false && "Can't insert move between two stack slot!");
  }
}

void MoveResolver::blockRegisters(LiveIntervalIdem *it) {
  if (renamer->isAssignedPhyReg(it)) {
    unsigned reg = renamer->getAssignedPhyReg(it);
    assert((multipleReadsAllowed || registerBlocked[reg] == 0) &&
    "register already marked as used");
    setRegisterBlocked(reg, 1);
  }
}

void MoveResolver::unblockRegisters(LiveIntervalIdem *it) {
  if (renamer->isAssignedPhyReg(it)) {
    unsigned reg = renamer->getAssignedPhyReg(it);
    assert(getRegisterBlocked(reg) > 0 && "register already marked as unused");
    setRegisterBlocked(reg, -1);
  }
}

void MoveResolver::setRegisterBlocked(unsigned reg, int direction) {
  assert(direction == 1 || direction == -1);
  assert(TargetRegisterInfo::isPhysicalRegister(reg));
  registerBlocked[reg] += direction;
}

int MoveResolver::getRegisterBlocked(unsigned reg) {
  return registerBlocked[reg];
}

bool MoveResolver::isSafeToProcessMove(LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt) {
  assert(renamer->isAssignedPhyReg(srcIt));
  unsigned srcReg = renamer->getAssignedPhyReg(srcIt);

  if (renamer->isAssignedPhyReg(dstIt)) {
    unsigned dstReg = renamer->getAssignedPhyReg(dstIt);
    if (getRegisterBlocked(dstReg) > 1 || (getRegisterBlocked(dstReg) == 1 || dstReg != srcReg))
      return false;
  }
  return tri;
}