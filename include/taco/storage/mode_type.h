#ifndef TACO_MODE_TYPE_H
#define TACO_MODE_TYPE_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>
#include <map>

#include "taco/format.h"
#include "taco/ir/ir.h"

namespace taco {

class IteratorImpl;
class ModeTypeImpl;
class ModeTypePack;
class ModePack;

namespace old {
class Iterators;
}


/// One of the modes of a tensor.
class Mode {
public:
  /// Construct a tensor mode.
  Mode(ir::Expr tensor, size_t level, Dimension size,
       const ModePack* pack, size_t packLoc, ModeType prevModeType);

  /// Retrieve the name of the tensor mode.
  std::string getName() const;

  /// Retrieve the tensor that contains the mode.
  ir::Expr getTensorExpr() const;

  /// Retrieve the level of this mode in its the mode hierarchy.  The first
  /// mode in a mode hierarchy is at level 1, and level 0 is the root level.
  size_t getLevel() const;

  /// Retrieve the size of the tensor mode.
  Dimension getSize() const;

  /// Retrieve the pack the mode partakes in.
  const ModePack* getPack() const;

  /// Retrieve the location of the mode in its mode pack.
  size_t getPackLocation() const;

  /// Retrieve the mode type of the parent level in the mode hierarchy.
  ModeType getParentModeType() const;

  /// Store temporary variables that may be needed to access or modify a mode
  /// @{
  ir::Expr getVar(std::string varName) const;
  bool     hasVar(std::string varName) const;
  void     addVar(std::string varName, ir::Expr var);
  /// @}

private:
  struct Content;
  std::shared_ptr<Content> content;
};


/// A mode pack consists of tensor modes that share the same physical arrays 
/// (e.g., modes of an array-of-structs COO tensor).
class ModePack {
public:
  /// Returns number of tensor modes belonging to mode pack.
  size_t getSize() const;

  /// Returns arrays shared by tensor modes.
  ir::Expr getArray(size_t i) const;

private:
  std::vector<Mode> modes;
  std::vector<ModeType> modeTypes;

  friend class old::Iterators;
  friend class Mode;
};


class ModeTypeImpl {
public:
  ModeTypeImpl() = delete;
  ModeTypeImpl(std::string name, bool isFull, bool isOrdered,
               bool isUnique, bool isBranchless, bool isCompact,
               bool hasCoordValIter, bool hasCoordPosIter, bool hasLocate,
               bool hasInsert, bool hasAppend);

  virtual ~ModeTypeImpl() {}

  /// Instantiates a variant of the mode type with differently configured 
  /// properties
  virtual ModeType
  copy(const std::vector<ModeType::Property>& properties) const = 0;


  /// Level functions that implement coordinate value iteration.
  /// @{
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getCoordIter(const std::vector<ir::Expr>& i, Mode& mode) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getCoordAccess(const ir::Expr& pPrev, const std::vector<ir::Expr>& i,
                 Mode& mode) const;
  /// @}


  /// Level functions that implement coordinate position iteration.
  /// @{
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getPosIter(const ir::Expr& pPrev, Mode& mode) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getPosAccess(const ir::Expr& p, const std::vector<ir::Expr>& i,
               Mode& mode) const;
  /// @}


  /// Level function that implements locate capability.
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getLocate(const ir::Expr& pPrev, const std::vector<ir::Expr>& i,
            Mode& mode) const;


  /// Level functions that implement insert capabilitiy.
  /// @{
  virtual ir::Stmt
  getInsertCoord(const ir::Expr& p, const std::vector<ir::Expr>& i,
                 Mode& mode) const;

  virtual ir::Expr getSize(Mode& mode) const;

  virtual ir::Stmt
  getInsertInitCoords(const ir::Expr& pBegin, const ir::Expr& pEnd,
                      Mode& mode) const;

  virtual ir::Stmt
  getInsertInitLevel(const ir::Expr& szPrev, const ir::Expr& sz,
                     Mode& mode) const;

  virtual ir::Stmt
  getInsertFinalizeLevel(const ir::Expr& szPrev, const ir::Expr& sz,
                         Mode& mode) const;
  /// @}

  
  /// Level functions that implement append capabilitiy.
  /// @{
  virtual ir::Stmt
  getAppendCoord(const ir::Expr& p, const ir::Expr& i, Mode& mode) const;

  virtual ir::Stmt
  getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin,
                 const ir::Expr& pEnd, Mode& mode) const;

  virtual ir::Stmt
  getAppendInitEdges(const ir::Expr& pPrevBegin, const ir::Expr& pPrevEnd,
                     Mode& mode) const;

  virtual ir::Stmt
  getAppendInitLevel(const ir::Expr& szPrev, const ir::Expr& sz,
                     Mode& mode) const;

  virtual ir::Stmt
  getAppendFinalizeLevel(const ir::Expr& szPrev, const ir::Expr& sz,
                         Mode& mode) const;
  /// @}


  /// Returns arrays associated with a tensor mode
  virtual ir::Expr getArray(size_t idx, const Mode& mode) const = 0;


  const std::string name;

  const bool isFull;
  const bool isOrdered;
  const bool isUnique;
  const bool isBranchless;
  const bool isCompact;

  const bool hasCoordValIter;
  const bool hasCoordPosIter;
  const bool hasLocate;
  const bool hasInsert;
  const bool hasAppend;
};

}
#endif

