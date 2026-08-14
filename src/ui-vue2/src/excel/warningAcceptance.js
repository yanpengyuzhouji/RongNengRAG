export function uniqueWarningCodes(warnings = []) {
  return [...new Set(warnings.map((warning) => warning?.code).filter(Boolean))]
}

export function acceptedWarningCodeCount(warnings = [], acceptedCodes = []) {
  const accepted = new Set(acceptedCodes)
  return uniqueWarningCodes(warnings).filter((code) => accepted.has(code)).length
}

export function areAllWarningCodesAccepted(warnings = [], acceptedCodes = []) {
  const accepted = new Set(acceptedCodes)
  return uniqueWarningCodes(warnings).every((code) => accepted.has(code))
}
