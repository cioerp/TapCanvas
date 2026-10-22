import { createHash } from 'node:crypto';
import { isJsonObject, type JsonObject, type RemoteToolDefinition } from './contracts.js';
import { validateJsonSchemaStructure, type JsonSchemaStructuralIssue } from './json-schema-structural-validator.js';
import type { ReferenceFactObservation } from '../../../../packages/schemas/json-schema-relations/reference-facts.mjs';

export const STRUCTURED_OUTPUT_TOOL = 'submit_structured_output';
export type StructuredSubmission = Readonly<{ value: JsonObject; contractHash: string }>;
const strings = (value: unknown): string[] => Array.isArray(value)
  ? value.filter((item): item is string => typeof item === 'string') : [];

/** Compile only explicit structural constraints; creative decisions stay with Harness. */
export function structuredOutputSchema(contract: JsonObject): JsonObject {
  const properties: JsonObject = {};
  const required = new Set<string>();
  const field = (name: string, schema: JsonObject): void => {
    properties[name] = { ...(isJsonObject(properties[name]) ? properties[name] : {}), ...schema };
    required.add(name);
  };
  for (const [key, type] of [['requiredStringFields', 'string'], ['requiredNumberFields', 'number'],
    ['requiredObjectFields', 'object'], ['requiredArrayFields', 'array']] as const) {
    for (const name of strings(contract[key])) field(name, { type, ...(type === 'string' ? { minLength: 1 } : {}) });
  }
  if (typeof contract.requiredArrayField === 'string') {
    field(contract.requiredArrayField, { type: 'array', minItems: typeof contract.minimumArrayLength === 'number' ? contract.minimumArrayLength : 1,
      ...(typeof contract.expectedArrayLength === 'number' ? { minItems: contract.expectedArrayLength, maxItems: contract.expectedArrayLength } : {}),
      ...(isJsonObject(contract.itemObject) ? { items: structuredOutputSchema(contract.itemObject) } : {}) });
  }
  for (const [key, constraints] of [['exactStringFields', 'const'], ['exactNumberFields', 'const'],
    ['stringAllowedValues', 'enum']] as const) {
    if (isJsonObject(contract[key])) for (const [name, value] of Object.entries(contract[key])) field(name, { [constraints]: value });
  }
  if (isJsonObject(contract.expectedArrayLengths)) for (const [name, length] of Object.entries(contract.expectedArrayLengths)) {
    field(name, { type: 'array', minItems: length, maxItems: length });
  }
  const arrayItems = (name: string): JsonObject => {
    const array = isJsonObject(properties[name]) ? properties[name] : { type: 'array' };
    const items = isJsonObject(array.items) ? array.items : { type: 'object', properties: {} };
    array.items = items;
    properties[name] = array;
    return items;
  };
  for (const [key, itemKey] of [['arrayItemRequiredStringFields', 'requiredStringFields'],
    ['arrayItemExactStringFields', 'exactStringFields'], ['arrayItemExactNumberFields', 'exactNumberFields'],
    ['arrayItemAllowedFields', 'allowedFields'], ['arrayItemNumberAllowedValues', 'stringAllowedValues']] as const) {
    if (!isJsonObject(contract[key])) continue;
    for (const [name, value] of Object.entries(contract[key])) {
      const items = arrayItems(name);
      const extra = structuredOutputSchema({ [itemKey]: value });
      const existing = isJsonObject(items.properties) ? items.properties : {};
      items.properties = { ...existing, ...(isJsonObject(extra.properties) ? extra.properties : {}) };
      items.required = [...new Set([...strings(items.required), ...strings(extra.required)])];
      if (extra.additionalProperties === false) items.additionalProperties = false;
    }
  }
  const allowed = strings(contract.allowedFields ?? contract.allowedTopLevelFields);
  for (const name of allowed) if (!properties[name]) properties[name] = {};
  const schema: JsonObject = { type: 'object', properties, required: [...required], ...(allowed.length ? { additionalProperties: false } : {}) };
  if (!isJsonObject(contract.jsonSchema)) return schema;
  const { $defs, definitions, ...artifactSchema } = contract.jsonSchema;
  return { allOf: [schema, artifactSchema], type: 'object',
    ...(isJsonObject($defs) ? { $defs } : {}),
    ...(isJsonObject(definitions) ? { definitions } : {}) };
}

export function structuredOutputTool(contract: JsonObject): RemoteToolDefinition {
  const artifactSchema = structuredOutputSchema(contract);
  const { $defs, definitions, ...outputSchema } = artifactSchema;
  return { name: STRUCTURED_OUTPUT_TOOL,
    description: 'Submit the exact JSON artifact required by this workflow node. Structural failures return actionable issues: repair the candidate and call this tool again in the same Harness turn. After acceptance stop authoring; the Bridge returns the accepted artifact verbatim.',
    parameters: { type: 'object', properties: { output: outputSchema }, required: ['output'], additionalProperties: false,
      ...(isJsonObject($defs) ? { $defs } : {}), ...(isJsonObject(definitions) ? { definitions } : {}) } };
}

/**
 * 传输层字符串化修复：部分模型（如 qwen3.8-27b 生成长结构化产物时）
 * 把 output 作为 JSON 字符串而非对象提交，甚至字符串本身是非法 JSON（对象提前闭合、字段被
 * 追加在根对象之后）。原行为下模型只收到 "$ must be object; received string"，无法定位
 * 问题，反复重发同一坏串 → structured_output_submission_missing 节点失败（实测：
 * 4 次提交全被拒，字符串在 char 723 处根对象提前闭合）。
 * 现在：合法 JSON 对象串 → 解析后按对象校验（接受）；非法 JSON 串 → 返回可行动的解析错误
 * 诊断（含 Node 的位置信息），明确告诉模型「直接提交对象、且文档必须是单个合法 JSON 值」。
 */
/**
 * 括号修复：部分模型（实测 qwen3.8-27b）生成 ~3K 字符嵌套 JSON 时
 * 系统性地在文档尾部放错闭合括号——要么多出一个 `]`（闭合数=开合数+1，解析错误恒在
 * len-2），要么 `}`/`]` 顺序颠倒。10 次重试同型缺陷，模型无法从 "Expected ',' or '}'"
 * 诊断中定位到字符级的括号错位 → 回合无有效提交即 structured_output_submission_missing。
 * 修复只做括号插入/删除（≤8 轮，按解析器位置 + 字符串感知的括号栈），修完仍走完整
 * schema 结构校验：修复不可能引入额外字段，结构不符仍会被拒。
 */
function scanBrackets(text: string, limit: number): {
  stack: Array<{ ch: "{" | "["; at: number }>;
  stray: number | null;
} {
  const stack: Array<{ ch: "{" | "["; at: number }> = [];
  let instr = false;
  let esc = false;
  let stray: number | null = null;
  for (let i = 0; i < Math.min(limit, text.length); i++) {
    const ch = text[i];
    if (instr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') instr = false;
      continue;
    }
    if (ch === '"') instr = true;
    else if (ch === "{" || ch === "[") stack.push({ ch, at: i });
    else if (ch === "}" || ch === "]") {
      const top = stack[stack.length - 1];
      const want = ch === "}" ? "{" : "[";
      if (!top || top.ch !== want) { stray = i; break; }
      stack.pop();
    }
  }
  return { stack, stray };
}

function repairJsonBrackets(text: string): unknown | null {
  let cur = text;
  for (let iter = 0; iter < 8; iter++) {
    try {
      return JSON.parse(cur);
    } catch (error) {
      const { stack, stray } = scanBrackets(cur, cur.length);
      if (stray !== null) {
        // 括号错位/多余：优先在错位点插入期望的闭合符（顺序颠倒型），否则删除该字符（多余型）。
        const top = stack[stack.length - 1]?.ch;
        const ch = cur[stray];
        if (top && ((ch === "]" && top === "{") || (ch === "}" && top === "["))) {
          const withInsert = cur.slice(0, stray) + (top === "{" ? "}" : "]") + cur.slice(stray);
          try { return JSON.parse(withInsert); } catch { /* 转删除 */ }
        }
        cur = cur.slice(0, stray) + cur.slice(stray + 1);
        continue;
      }
      if (stack.length > 0) {
        // 文档截断：整串括号自洽但缺尾部闭合符 → 按栈序补齐。
        const closers = [...stack].reverse().map((o) => (o.ch === "{" ? "}" : "]")).join("");
        const candidate = cur + closers;
        try { return JSON.parse(candidate); } catch { return null; }
      }
      // 括号全平衡但解析失败：解析器在文档中部报位置（内容错误，如键值对落在数组里），
      // 机械修复无法安全表达模型意图 → 交回诊断，让模型自己改结构。
      const m = (error instanceof Error ? error.message : String(error)).match(/position (\d+)/);
      if (m && parseInt(m[1], 10) < cur.length) return null;
      // 尾部根值已闭合后跟了垃圾：截到首个平衡点。
      for (let i = 0; i < cur.length; i++) {
        const mid = scanBrackets(cur, i + 1);
        if (mid.stray === null && mid.stack.length === 0) {
          const candidate = cur.slice(0, i + 1);
          try { return JSON.parse(candidate); } catch { /* 继续 */ }
          break;
        }
      }
      return null;
    }
  }
  try { return JSON.parse(cur); } catch { return null; }
}

function coerceStringifiedArtifact(value: unknown): {
  value: unknown;
  issue?: JsonSchemaStructuralIssue;
} {
  if (typeof value !== "string") return { value };
  const text = value.trim();
  if (!text.startsWith("{") && !text.startsWith("[")) return { value };
  try {
    const parsed: unknown = JSON.parse(text);
    return isJsonObject(parsed) ? { value: parsed } : { value };
  } catch (error) {
    const repaired = text.startsWith("{") ? repairJsonBrackets(text) : null;
    if (repaired !== null && isJsonObject(repaired)) return { value: repaired };
    const detail = error instanceof Error ? error.message : String(error);
    return {
      value,
      issue: {
        path: "$",
        keyword: "type",
        message:
          `output 收到了 JSON 字符串而不是对象，且该字符串不是合法 JSON（解析失败：${detail}）。` +
          "请把 output 直接作为 JSON 对象提交（不要序列化成字符串），并确保整份文档是单个合法 JSON 值（不要提前闭合根对象、不要在根对象外追加字段），然后重新调用本工具。",
      },
    };
  }
}

export function inspectStructuredSubmission(contract: JsonObject, value: unknown): {
  submission: StructuredSubmission | null; issues: readonly JsonSchemaStructuralIssue[];
  observations: readonly ReferenceFactObservation[];
} {
  const { value: coerced, issue } = coerceStringifiedArtifact(value);
  const observations: ReferenceFactObservation[] = [];
  const issues = issue
    ? [issue]
    : validateJsonSchemaStructure({ schema: structuredOutputSchema(contract), value: coerced, observations });
  if (!isJsonObject(coerced) || issues.length) return { submission: null, issues, observations };
  return { submission: { value: coerced, contractHash: `sha256:${createHash('sha256').update(JSON.stringify(contract)).digest('hex')}` }, issues, observations };
}
