# IRH v21.4 Compliance System - User Guide

## 🎯 What You Need to Know

This repository now has a **comprehensive compliance system** that ensures every contribution meets the rigorous theoretical standards of IRH v21.4. This guide explains what you need to do as a contributor.

---

## ⚡ Quick Start (5 Minutes)

### 1. Read the Standards (MUST DO)
```bash
# Open and read this file:
cat .github/THEORETICAL_CORRESPONDENCE_MANDATE.md
```

**Key takeaways:**
- ✅ Complete formulas (no shortcuts)
- ✅ Manuscript citations (IRH v21.4 Part 1/2, §X.Y, Eq. Z)
- ✅ Transparency logs (all computations must emit provenance)
- ❌ No hardcoded constants
- ❌ No black box code

### 2. Check Your Code
```bash
# Before committing, run:
python scripts/verify_compliance.py --verbose
```

### 3. Fix Violations
If the script shows violations, fix them following examples in:
- `.github/COMPLIANCE_QUICK_REFERENCE.md` - Common fixes
- `.github/THEORETICAL_CORRESPONDENCE_MANDATE.md` - Code templates

### 4. Open PR
- Template loads automatically with checklist
- Complete all mandatory items
- CI will verify and comment

---

## 📚 Documentation Hierarchy

### For Beginners
**Start here:** `.github/COMPLIANCE_QUICK_REFERENCE.md`
- Quick fixes for common violations
- Copy-paste templates
- 5-minute read

### For Daily Use
**Bookmark this:** `.github/COMPLIANCE_SYSTEM_README.md`
- Complete system overview
- All tools explained
- Critical issues summary

### For Deep Understanding
**Read when time permits:**
1. `.github/THEORETICAL_CORRESPONDENCE_MANDATE.md` - The standards (19KB)
2. `.github/COMPREHENSIVE_AUDIT_REPORT.md` - Current gaps (18KB)
3. `.github/MANDATORY_AUDIT_PROTOCOL.md` - Audit procedures (11KB)

---

## 🛠️ Tools Available

### 1. Compliance Verification Script
**Location:** `scripts/verify_compliance.py`

**What it does:**
- Checks all your code against IRH v21.4 standards
- Identifies violations before you commit
- Generates reports for tracking

**How to use:**
```bash
# Basic check
python scripts/verify_compliance.py

# Detailed output
python scripts/verify_compliance.py --verbose

# Save report
python scripts/verify_compliance.py --report compliance_report.json
```

**When to use:**
- ✅ Before every commit
- ✅ After adding new functions
- ✅ When fixing violations
- ✅ Before opening a PR

### 2. PR Template
**Location:** `.github/pull_request_template.md`

**What it does:**
- Loads automatically when you open a PR
- Provides comprehensive checklist
- Documents rejection criteria

**How to use:**
- Fill out all sections
- Check all mandatory items
- Add context in "Additional Notes"

### 3. CI/CD Workflow
**Location:** `.github/workflows/compliance_check.yml`

**What it does:**
- Runs automatically on every PR
- Posts compliance results as comment
- Blocks merge if violations found

**What you see:**
- ✅ Green check: Compliant, ready to merge
- ❌ Red X: Violations found, need fixes
- Comment on PR with detailed results

---

## 🚨 Top 5 Things That Will Get Your PR Rejected

### 1. ❌ Missing Manuscript Citations
```python
# This will be REJECTED:
def compute_mass(K_f):
    return 0.511 * K_f
```

```python
# This will be ACCEPTED:
def compute_mass(K_f):
    """
    Theoretical Reference:
        IRH v21.4 Part 1, §3.2.1, Eq. 3.6
    
    Formula (Complete):
        m_f = [full formula]
    """
    return prefactor * K_f
```

### 2. ❌ Hardcoded Physical Constants
```python
# REJECTED - Where does 137.036 come from?
ALPHA_INVERSE = 137.035999084
```

```python
# ACCEPTED - Computed from first principles
def compute_alpha_inverse():
    """IRH v21.4 Part 1, §3.2.2, Eq. 3.4"""
    return (4 * pi**2 * GAMMA_STAR) / LAMBDA_STAR * corrections

ALPHA_INVERSE = compute_alpha_inverse()
```

### 3. ❌ Simplified Formulas
```python
# REJECTED - Missing terms
m_f = prefactor * sqrt(K_f)
```

```python
# ACCEPTED - Complete formula per Eq. 3.6
m_f = R_Y * sqrt(2) * K_f * sqrt(lambda_star) * \
      sqrt(mu_star/lambda_star) * planck_length_inverse
```

### 4. ❌ Black Box Code
```python
# REJECTED - What does this compute?
result = optimizer.optimize(data)
return result
```

```python
# ACCEPTED - Transparent with provenance
from src.logging import TransparencyEngine, FULL
engine = TransparencyEngine(verbosity=FULL)
engine.info("Computing X", reference="§X.Y, Eq. Z")
result = optimizer.optimize(data)
engine.value("result", result)
return result
```

### 5. ❌ No Tests
```python
# REJECTED - No validation
def new_physics_function():
    pass
```

```python
# ACCEPTED - Tested against theory
def new_physics_function():
    """IRH v21.4 Part 1, §X.Y, Eq. Z"""
    pass

# tests/test_module.py
def test_new_physics_function():
    """Verify Eq. Z implementation."""
    result = new_physics_function()
    assert np.isclose(result, expected_from_theory)
```

---

## ✅ Checklist Before Every Commit

```markdown
- [ ] Read .github/THEORETICAL_CORRESPONDENCE_MANDATE.md
- [ ] All functions cite IRH v21.4 Part 1/2, §X.Y, Eq. Z
- [ ] Complete formulas (no missing terms)
- [ ] No hardcoded constants (all computed or justified)
- [ ] TransparencyEngine integrated where needed
- [ ] Tests added and passing
- [ ] Run: python scripts/verify_compliance.py
- [ ] All violations fixed
```

**Save this checklist and use it before every commit!**

---

## 🎓 Learning Path

### Day 1: Understanding (30 minutes)
1. Read `.github/COMPLIANCE_QUICK_REFERENCE.md`
2. Run `python scripts/verify_compliance.py` on existing code
3. Look at examples of violations and fixes

### Day 2: Practicing (1 hour)
1. Pick one violation from compliance report
2. Fix it using examples from mandate
3. Run compliance check to verify
4. Commit the fix

### Day 3: Mastering (2 hours)
1. Write a new function with full compliance
2. Add transparency logs
3. Write tests
4. Pass compliance check on first try
5. Open a PR

**After Day 3, you're a compliance champion!** 🏆

---

## 🆘 When You Need Help

### Compliance Script Fails
1. Read error messages carefully
2. Check `.github/COMPLIANCE_QUICK_REFERENCE.md` for fixes
3. Look for similar code in compliant modules
4. Fix and re-run

### Not Sure If Compliant
1. Check examples in `.github/THEORETICAL_CORRESPONDENCE_MANDATE.md`
2. Look at recent PRs that passed CI
3. Ask in issue with "COMPLIANCE" tag

### Test Failures
1. Check theoretical expectations
2. Verify formula implementation
3. Use TransparencyEngine for debugging
4. Check dimensional consistency

### PR Blocked by CI
1. Read CI comment on PR
2. Fix listed violations
3. Push fixes
4. CI will re-run automatically

---

## 📈 Your Impact

**Every compliant function you write:**
- ✅ Moves IRH toward 100% theoretical fidelity
- ✅ Ensures reproducible science
- ✅ Enables verifiable predictions
- ✅ Maintains publication quality

**You're not just writing code, you're building a computational engine of reality!**

---

## 🎯 Current Progress

**Baseline:** 35% compliance (959 violations)
**Target:** 100% compliance
**Timeline:** 10 weeks
**Your contribution:** Reduces violation count, increases fidelity

**Track progress:**
```bash
# See current status
python scripts/verify_compliance.py | grep "Status:"

# Compare to last week
git log --grep="compliance" --oneline
```

---

## 🔗 Quick Links

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [Quick Reference](.github/COMPLIANCE_QUICK_REFERENCE.md) | Daily use | Every day |
| [System README](.github/COMPLIANCE_SYSTEM_README.md) | Complete overview | Once |
| [Mandate](.github/THEORETICAL_CORRESPONDENCE_MANDATE.md) | Standards | Before coding |
| [Audit Report](.github/COMPREHENSIVE_AUDIT_REPORT.md) | Current gaps | When fixing issues |
| [PR Template](.github/pull_request_template.md) | Checklist | When opening PR |

---

## 💡 Pro Tips

### Tip 1: Run Compliance Early
Don't wait until end of development. Check after every function.

### Tip 2: Use Templates
Copy examples from mandate instead of writing from scratch.

### Tip 3: Add Transparency From Start
Easier to add while coding than retrofitting later.

### Tip 4: Test First
Write test showing theoretical expectation, then implement to pass.

### Tip 5: Ask Questions
Better to ask before coding than fix violations after.

---

## 🤝 We're Here to Help

This system exists to **help you write better code**, not to be a burden.

**Remember:**
- Standards are documented
- Tools are automated
- Examples are provided
- Help is available

**Together, we're building something extraordinary:**
A computational framework that doesn't approximate reality—it computes it from first principles with complete transparency.

---

## 📞 Contact

**Questions?** Open an issue with tag:
- `COMPLIANCE` - Tool or process questions
- `THEORY CLARIFICATION` - Manuscript questions
- `HELP WANTED` - Need assistance

**Found a bug in compliance tools?** Open issue with:
- Steps to reproduce
- Expected vs. actual behavior
- Output of `python scripts/verify_compliance.py`

---

## 🚀 Let's Build Together

**Your mission:** Write code that meets the highest standards of theoretical rigor.

**Our mission:** Provide tools that make this easy and automatic.

**Our shared goal:** Transform IRH v21.4 into a verified computational engine of reality.

---

**Welcome to the compliance system. Let's build something incredible!** 🌟

---

*Last Updated: December 2025*
*Version: IRH v21.4 Compliance System v1.0*
*Status: ACTIVE*
