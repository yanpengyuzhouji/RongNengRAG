import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { compileScript, compileStyle, compileTemplate, parse } from '@vue/compiler-sfc'

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const sourceRoot = path.join(projectRoot, 'src')

function vueFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) return vueFiles(target)
    return entry.isFile() && entry.name.endsWith('.vue') ? [target] : []
  })
}

for (const filename of vueFiles(sourceRoot)) {
  const source = fs.readFileSync(filename, 'utf8')
  const relative = path.relative(projectRoot, filename)
  const { descriptor, errors } = parse(source, { filename: relative })
  if (errors.length) throw errors[0]

  let bindings = {}
  if (descriptor.script || descriptor.scriptSetup) {
    const compiled = compileScript(descriptor, { id: relative })
    bindings = compiled.bindings
  }
  if (descriptor.template) {
    const result = compileTemplate({
      source: descriptor.template.content,
      filename: relative,
      id: relative,
      compilerOptions: { bindingMetadata: bindings },
    })
    if (result.errors.length) throw result.errors[0]
  }
  for (const style of descriptor.styles) {
    const result = compileStyle({
      source: style.content,
      filename: relative,
      id: relative,
      scoped: style.scoped,
    })
    if (result.errors.length) throw result.errors[0]
  }
}

console.log(`SFC compile: ${vueFiles(sourceRoot).length} files OK`)
