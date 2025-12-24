# IRH v21.4 Compliance Quick Reference

## 🚀 Quick Start (5 Minutes)

### Before You Code
```bash
# 1. Read the mandate
cat .github/THEORETICAL_CORRESPONDENCE_MANDATE.md

# 2. Check audit report for area you're working on
grep -A 10 "CRITICAL" .github/COMPREHENSIVE_AUDIT_REPORT.md
```

### While You Code
```python
# ALWAYS include in functions:
"""
Theoretical Reference:
    IRH v21.4 Part 1, §X.Y, Eq. Z
    
Formula (Complete):
    [full formula with ALL terms]
"""

# ALWAYS use transparency:
from src.logging import TransparencyEngine, FULL
engine = TransparencyEngine(verbosity=FULL)
engine.info("Computing X", reference="§X.Y, Eq. Z")
```

### Before You Commit
```bash
# Run compliance check
python scripts/verify_compliance.py --verbose

# If non-compliant, fix violations then recheck
```

---

## ⚠️ Top 5 Violations to Avoid

### 1. ❌ Missing Manuscript Citations
```python
# WRONG
def compute_mass(K_f):
    return 0.511 * K_f
```

```python
# RIGHT
def compute_mass(K_f):
    """
    Theoretical Reference:
        IRH v21.4 Part 1, §3.2.1, Eq. 3.6
    """
    return prefactor * K_f  # Complete formula
```

### 2. ❌ Hardcoded Physical Constants
```python
# WRONG
ALPHA_INVERSE = 137.035999084  # Where from?
```

```python
# RIGHT
def compute_alpha_inverse():
    """
    Theoretical Reference:
        IRH v21.4 Part 1, §3.2.2, Eq. 3.4
        
    Formula (Complete):
        α⁻¹ = (4π²γ̃*/λ̃*) × [1 + corrections...]
    """
    # Compute from first principles
    return leading_order * correction_factor
```

### 3. ❌ Simplified Formulas
```python
# WRONG - Missing RG running
m_f = prefactor * sqrt(K_f) * higgs_vev
```

```python
# RIGHT - Complete formula
m_f = R_Y * sqrt(2) * K_f * sqrt(lambda_star) * sqrt(mu_star/lambda_star) * l_planck_inv
```

### 4. ❌ Black Box Computations
```python
# WRONG
result = optimizer.optimize(data)
return result
```

```python
# RIGHT
engine = TransparencyEngine(verbosity=FULL)
engine.info("Optimizing...", reference="§X.Y")
engine.step("Step 1: Initialize")
result = optimizer.optimize(data)
engine.value("result", result, uncertainty=1e-12)
return result
```

### 5. ❌ No Tests
```python
# WRONG - No validation
def new_function():
    pass
```

```python
# RIGHT - With tests
def new_function():
    """Theoretical Reference: §X.Y, Eq. Z"""
    pass

# tests/test_module.py
def test_new_function():
    """Verify Eq. Z implementation."""
    assert new_function() == expected_from_theory
```

---

## 📋 Compliance Checklist (Copy-Paste)

```markdown
Before committing:
- [ ] All functions cite IRH v21.4 Part 1/2, §X.Y, Eq. Z
- [ ] Complete formulas (no missing terms)
- [ ] No hardcoded constants (all computed or justified)
- [ ] TransparencyEngine integrated
- [ ] Tests added and passing
- [ ] `python scripts/verify_compliance.py` passes
```

---

## 🔧 Common Fixes

### Fix: Add Manuscript Citation
```python
# Add to docstring:
"""
Theoretical Reference:
    IRH v21.4 Part 1, §3.2.2, Eq. 3.4
    Appendix E.4 for derivation details
"""
```

### Fix: Replace Hardcoded Constant
```python
# Before:
ALPHA_INVERSE = 137.035999084

# After:
def compute_alpha_inverse():
    """IRH v21.4 Part 1, §3.2.2, Eq. 3.4"""
    return (4 * pi**2 * GAMMA_STAR) / LAMBDA_STAR * correction_factor

ALPHA_INVERSE = compute_alpha_inverse()  # Computed, not hardcoded
```

### Fix: Add Transparency
```python
# Add at function start:
from src.logging import TransparencyEngine, FULL

engine = TransparencyEngine(verbosity=FULL)
engine.info("Computing X", reference="§X.Y, Eq. Z")

# Add throughout:
engine.step("Step 1: ...")
engine.value("intermediate", value, uncertainty=1e-12)

# Add at end:
engine.passed("Computation complete")
```

### Fix: Complete Formula
```python
# Before (simplified):
mass = prefactor * K_f

# After (complete per Eq. 3.6):
mass = R_Y * sqrt(2) * K_f * sqrt(lambda_star) * \
       sqrt(mu_star/lambda_star) * planck_length_inverse
```

---

## 🎯 Priority Order

### If you're fixing existing code:
1. **CRITICAL-5** ✅ - Already done (Transparency Engine)
2. **CRITICAL-1** 🔴 - Fermion masses (highest impact)
3. **CRITICAL-2** 🔴 - Alpha inverse (flagship prediction)
4. **CRITICAL-4** 🔴 - Topological complexity (foundational)
5. **CRITICAL-3** 🔴 - Beta functions (RG flow basis)

### If you're adding new code:
1. Read mandate first
2. Use templates from mandate
3. Add transparency from start
4. Verify compliance before commit

---

## 🆘 Emergency Contact

### If compliance check fails:
1. Read error messages carefully
2. Check `.github/THEORETICAL_CORRESPONDENCE_MANDATE.md` for examples
3. Fix violations listed
4. Re-run: `python scripts/verify_compliance.py`

### If you're unsure:
1. Look for similar functions in codebase
2. Check audit report for guidance
3. Open issue with "COMPLIANCE" tag

### If test fails:
1. Check theoretical expectations
2. Verify formula implementation
3. Add debugging with TransparencyEngine
4. Check dimensional consistency

---

## 📚 Essential Reading (Priority Order)

1. **MUST READ** (15 min):
   - `.github/THEORETICAL_CORRESPONDENCE_MANDATE.md`
   
2. **SHOULD READ** (20 min):
   - `.github/COMPREHENSIVE_AUDIT_REPORT.md` (focus on your area)
   
3. **GOOD TO READ** (10 min):
   - `AUDIT_SUMMARY_FOR_USER.md`
   - `.github/COMPLIANCE_SYSTEM_README.md`

---

## 🔍 Quick Diagnostic

### Is my code compliant?
```bash
# Run this:
python scripts/verify_compliance.py --verbose | grep "✗ VIOLATION"

# If no output: ✅ Compliant
# If output shows violations: ❌ Fix and rerun
```

### Does my function need citations?
**YES if:**
- It computes a physical observable
- It implements an equation from manuscript
- It's public API (not starting with `_`)

**NO if:**
- It's a utility function (e.g., `_to_bytes()`)
- It's pure infrastructure (e.g., logging)
- It's a test helper

### Does it need transparency?
**YES if:**
- It computes observables (α, masses, w₀, etc.)
- It solves equations (RG flow, Wetterich, etc.)
- It performs numerical integration
- Users need to understand what it's doing

**NO if:**
- It's a simple getter/setter
- It's pure data transformation
- It's already wrapped by transparent parent

---

## 💡 Pro Tips

### Tip 1: Start with Tests
```python
# Write test FIRST showing what theory predicts
def test_alpha_at_fixed_point():
    """α⁻¹ should be 137.035999084 per Eq. 3.4"""
    assert np.isclose(compute_alpha(), 137.035999084, rtol=1e-10)

# THEN implement to pass test
```

### Tip 2: Use Engine Early
```python
# Add transparency WHILE developing, not after
engine = TransparencyEngine(verbosity=FULL)
# You'll catch bugs faster!
```

### Tip 3: Check Examples
```python
# Look at compliant implementations:
# - src/logging/transparency_engine.py (itself)
# - Examples in MANDATE.md
# - Recent PRs that passed compliance
```

### Tip 4: Run Often
```bash
# Don't wait until end to check compliance
python scripts/verify_compliance.py

# Run after every function you add
```

---

## 📈 Success Metrics

**Your PR is ready when:**
- ✅ `verify_compliance.py` passes
- ✅ All tests pass
- ✅ PR template checklist complete
- ✅ Reviewer approves

**You're a compliance champion when:**
- ✅ Zero violations in your code
- ✅ Transparency logs comprehensive
- ✅ Tests verify theoretical properties
- ✅ Documentation complete

---

## 🎓 Learning Path

### Day 1: Understanding
- Read mandate
- Run compliance check on existing code
- Identify patterns

### Day 2: Practicing
- Fix one violation
- Add one citation
- Run tests

### Day 3: Mastering
- Write new function with full compliance
- Add transparency logs
- Pass compliance check first try

---

**Remember:** Compliance is not a burden, it's the foundation of scientific rigor.

**The goal:** Transform IRH from approximation to precision computational engine.

**Your contribution:** Every compliant function moves us closer to 100% theoretical fidelity.

---

*Keep this file bookmarked for daily reference.*
*When in doubt, read the mandate.*
*When stuck, check the audit report.*
*When ready, run verify_compliance.py.*

**Let's build a computational engine of reality together! 🚀**
