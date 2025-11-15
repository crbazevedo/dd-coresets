# Proposals for README Opening Line

## Current (Updated)
**"We need to compress a large dataset into a small weighted subset that preserves all clusters, modes, and marginal distributions—not just global statistics."**

## Analysis

### Why the original "30M to 500" is problematic:
1. **Too specific**: No evidence that 30M→500 is optimal or even tested
2. **Misleading**: Doesn't highlight DDC's actual advantages
3. **Ignores findings**: Our experiments show DDC excels in specific scenarios (clustered data, small k), not just any large dataset

### What we learned from experiments:
- DDC excels when: **clustered data**, **small k relative to n**, **spatial coverage matters**
- DDC preserves: **all clusters** (even small ones), **marginal distributions**, **mean** (57% better)
- DDC trade-off: Random may preserve **global covariance** better, but DDC preserves **structure**

## Alternative Proposals

### Option 1: Structure-focused (Current)
**"We need to compress a large dataset into a small weighted subset that preserves all clusters, modes, and marginal distributions—not just global statistics."**

**Pros**: 
- Highlights DDC's strengths (clusters, modes, marginals)
- Contrasts with "global statistics" (where Random may be better)
- General enough to apply to various sizes

**Cons**: 
- Slightly abstract
- Doesn't mention the "guarantee" aspect

---

### Option 2: Guarantee-focused
**"We need a small weighted subset that guarantees coverage of all clusters and modes, preserving the full distribution—not just global statistics."**

**Pros**:
- Emphasizes "guarantee" (key DDC advantage)
- Clear about what's preserved
- Still general

**Cons**:
- "Full distribution" might be too strong (it's an approximation)

---

### Option 3: Concrete but flexible
**"We need to compress millions of rows into hundreds of weighted points that preserve all clusters, modes, and marginal distributions—not just global statistics."**

**Pros**:
- More concrete (millions→hundreds)
- Still flexible (not tied to specific numbers)
- Clear scale

**Cons**:
- "Millions" might not apply to all use cases
- Still somewhat arbitrary

---

### Option 4: Problem-solution focused
**"Random sampling can miss important clusters and modes. We need a small weighted subset that guarantees spatial coverage and preserves marginal distributions—not just global covariance."**

**Pros**:
- Directly addresses the problem
- Mentions Random (common baseline)
- Highlights "guarantee" and "spatial coverage"
- Acknowledges trade-off (covariance)

**Cons**:
- Longer
- More negative framing

---

### Option 5: Use-case focused
**"When you need to compress a large dataset into a small coreset, DDC ensures all clusters are represented and marginal distributions are preserved—not just global statistics."**

**Pros**:
- Mentions DDC explicitly
- Clear use case
- Emphasizes "ensures" (guarantee)

**Cons**:
- Slightly less punchy

---

## Recommendation

**Option 1 (Current)** is the best balance:
- ✅ Accurate based on findings
- ✅ Highlights DDC's strengths
- ✅ General enough for various scenarios
- ✅ Clear contrast with "global statistics"
- ✅ Concise and punchy

**Alternative**: If we want to be more concrete, **Option 3** works well:
- "We need to compress millions of rows into hundreds of weighted points that preserve all clusters, modes, and marginal distributions—not just global statistics."

## Key Principles for the Opening Line

1. **Focus on DDC's strengths**: Clusters, modes, marginals, spatial coverage
2. **Acknowledge trade-offs**: Not just "better than Random", but "better at structure"
3. **Be general**: Don't tie to specific numbers unless we have evidence
4. **Be accurate**: Based on experimental findings
5. **Be clear**: What problem does DDC solve?

## Supporting Evidence

From `examples/NOTEBOOKS_SUMMARY.md`:
- **DDC excels at mean preservation**: 57% better than Random
- **DDC ensures spatial coverage**: 65.7% vs 30.3% coverage of largest cluster
- **DDC preserves marginals**: 25.5% better W1 with adaptive distances
- **DDC guarantees cluster representation**: All clusters covered, even small ones

From `docs/DDC_ADVANTAGE_CASES.md`:
- DDC superior when: well-defined clusters, small k, spatial coverage matters
- Random superior when: preserving exact global covariance is critical

