import assert from 'node:assert/strict'

import {
  acceptedWarningCodeCount,
  areAllWarningCodesAccepted,
  uniqueWarningCodes,
} from '../src/excel/warningAcceptance.js'

const warnings = [
  { code: 'FORMULAS', sheet_id: 'a' },
  { code: 'FORMULAS', sheet_id: 'b' },
  { code: 'MERGED_CELLS', sheet_id: 'b' },
]

assert.deepEqual(uniqueWarningCodes(warnings), ['FORMULAS', 'MERGED_CELLS'])
assert.equal(acceptedWarningCodeCount(warnings, ['FORMULAS']), 1)
assert.equal(areAllWarningCodesAccepted(warnings, ['FORMULAS']), false)
assert.equal(
  areAllWarningCodesAccepted(warnings, ['FORMULAS', 'MERGED_CELLS']),
  true,
)

// Old/stale accepted codes must not enable confirmation by merely matching a count.
assert.equal(
  areAllWarningCodesAccepted(warnings, ['OLD_A', 'OLD_B', 'OLD_C']),
  false,
)
assert.equal(areAllWarningCodesAccepted([], []), true)
