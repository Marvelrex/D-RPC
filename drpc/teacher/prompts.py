import json

SUPER_CORRECT_PROMPTS = """
Transform the solution of the following math problem into a step-by-step XML format, each step should
be enclosed within tags like ⟨Step1⟩ ⟨/Step1⟩. For each step enclosed within the tags, determine if this
step is challenging and tricky, if so, add detailed explanation and analysis enclosed within ⟨Key⟩ ⟨/Key⟩ in this step, as helpful annotations to make the student better understand this step correctly thus mastering
the solution. After all the reasoning steps, summarize the common solution and reasoning steps to help
him generalize to similar problems within ⟨Generalized⟩ ⟨/Generalized⟩. Finally present the final answer
enclosed within ⟨Answer⟩ ⟨/Answer⟩.
"""

CATEGORY_INTENT_PROMPT = """
Return ONLY a JSON object {"category": "<category>", "intent": "<intent>"} for the math question below. 

Definitions:
- category: Use exactly one core mathematical concept word to summarize question. 
- intent: The goal of calculate to solve the problem. Four words max.

Question: 
"""

GSM8K_FEW_SHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "Let's think step by step. There are 15 trees originally. After planting, there are 21. The number planted is 21 - 15 = 6. Answer:6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "Let's think step by step. Start with 3 cars. Add 2 more: 3 + 2 = 5. Answer:5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Let's think step by step. Leah has 32, her sister 42, total 32 + 42 = 74. They eat 35, so 74 - 35 = 39 remain. Answer:39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Let's think step by step. Jason starts with 20 and ends with 12. He gave away 20 - 12 = 8. Answer:8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Let's think step by step. Shawn had 5 toys. He gets 2 from mom and 2 from dad, adding 4. Total 5 + 4 = 9. Answer:9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "Let's think step by step. Start with 9 computers. From Monday to Thursday is 4 days, add 5 each day: 5 * 4 = 20. Total 9 + 20 = 29. Answer:29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Let's think step by step. Begin with 58. After losing 23: 58 - 23 = 35. After losing 2 more: 35 - 2 = 33. Answer:33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Let's think step by step. Each bagel costs 3, so 5 cost 5 * 3 = 15. Olivia started with 23, so 23 - 15 = 8 remain. Answer:8",
    ),
]

COT_FEW_SHOT_EXAMPLES = GSM8K_FEW_SHOT_EXAMPLES

SVAMP_COT_FEW_SHOT_EXAMPLES = [
    (
        "Liam read 12 pages on Monday and 15 pages on Tuesday. He wants to read 40 pages this week. How many pages does he still need to read?",
        "Let's think step by step. He read 12 + 15 = 27 pages. He needs 40 - 27 = 13 more pages. Answer:13",
    ),
    (
        "A shop sold 28 apples in the morning and 17 in the afternoon. They started with 60 apples. How many are left?",
        "Let's think step by step. Total sold 28 + 17 = 45. Leftover is 60 - 45 = 15 apples. Answer:15",
    ),
    (
        "Mia had $45. She bought a book for $18 and a notebook for $7. How much money does she have left?",
        "Let's think step by step. Total spent 18 + 7 = 25. Money left 45 - 25 = 20. Answer:20",
    ),
    (
        "There are 24 students. They sit in rows of 6. How many rows are there?",
        "Let's think step by step. 24 divided by 6 is 4 rows. Answer:4",
    ),
    (
        "A box has 36 candies. If 9 candies are given to each child, how many children get candies?",
        "Let's think step by step. 36 / 9 = 4 children. Answer:4",
    ),
]

STRATEGYQA_COT_ZERO_SHOT = """Use chain-of-thought reasoning ("Let's think step by step") to solve the problem.

Let's think step by step. Show concise reasoning and end with Answer:<bool>.
Question: {question}
"""

DEFAULT_NORMAL_FEW_SHOT_COUNT = len(GSM8K_FEW_SHOT_EXAMPLES)


NORMAL_BASE_INSTRUCTIONS = """
Use chain-of-thought reasoning ("Let's think step by step") to solve the problem.

Rules:
- Finish the last answer with the answer.
- Keep reasoning concise and focused on the calculation.
"""


def _format_normal_examples(examples: list[tuple[str, str]]) -> str:
    lines = []
    for question, answer in examples:
        lines.append(f"Q: {question}")
        lines.append(f"A: {answer}")
        lines.append("")
    return "\n".join(lines).strip()


def svamp_cot_few_shot(num_shots: int | None = None) -> str:
    """Few-shot CoT examples tailored to SVAMP-style arithmetic word problems."""
    examples = SVAMP_COT_FEW_SHOT_EXAMPLES
    count = len(examples) if num_shots is None else max(0, min(len(examples), int(num_shots)))
    if count <= 0:
        return "Let's think step by step and finish with Answer:<number>."
    return _format_normal_examples(examples[:count])


def normalize_part_three(num_shots: int | None = None) -> str:
    """Few-shot Chain-of-Thought instructions for the normal strategy."""
    if num_shots is None:
        count = DEFAULT_NORMAL_FEW_SHOT_COUNT
    else:
        count = max(0, int(num_shots))

    prompt_sections = []
    if count:
        selected_examples = GSM8K_FEW_SHOT_EXAMPLES[:count]
        examples_block = _format_normal_examples(selected_examples)
        prompt_sections.append(examples_block)
    return "\n\n".join(prompt_sections).strip()

PART_ONE_ROLE = """
You are a rigorous but concise math tutor.
You solve math problems carefully and explain your reasoning briefly and clearly.
Avoid unnecessary prose; show only the key steps needed to reach the answer
"""

PART_TWO_TASK = """
Your task:
Solve the question and return ONLY a valid JSON object.

Hard rules:
- Output JSON only (no markdown, no extra text).
- The JSON must have exactly two top-level keys: "rationale" and "ans".
- "ans" must be a numeric value (integer or decimal), not a string.
- Ensure valid JSON punctuation (quotes, commas, braces).
- Do not restate these rules in the output.
"""

Structured_CATEGORY_BASED_FREEFORM_PART_THREE = """
Output ONLY valid JSON in following format, and AVOID QUESTION SPECIFIC WORDING in reasoning_path keys:
{
  "route": {
    "difficulty": <1|2|3>,
    "budget": <Follow Budget Contract>,
    "reasoning_path": ["<DescriptiveStepName1>", "<DescriptiveStepName2>", ...]
  },
  "rationale": {
    "<DescriptiveStepName1>": "<short reasoning with derivation>",
    "<DescriptiveStepName2>": "<short reasoning with derivation>",
    ...
  },
  "ans": <numeric>
}

- Difficulty: How hard is this question (1-3).
- Budget: The maximum number of steps allowed.
"""

Structured_CATEGORY_BASED_FREEFORM_PART_THREE_DETAILED = """
Output ONLY valid JSON in following format, and AVOID QUESTION SPECIFIC WORDING in reasoning_path keys:
{
  "route": {
    "difficulty": <1|2|3>,
    "budget": <Follow Budget Contract>,
    "reasoning_path": ["<DescriptiveStepName1>", "<DescriptiveStepName2>", ...]
  },
  "rationale": {
    "<DescriptiveStepName1>": "Step1: <...> Step2: <...> Step3: <...>",
    "<DescriptiveStepName2>": "Step1: <...> Step2: <...>",
    ...
  },
  "ans": <numeric>
}

- Difficulty: How hard is this question (1-3).
- Budget: The maximum number of steps allowed.
- Rationale: For each reasoning_path entry, write a detailed multi-step explanation in a single string.
  Use Step1:, Step2:, Step3:, ... labels; choose as many substeps as needed (they do NOT count toward budget).
"""


Structured_RPB_PART_THREE = """
Output ONLY valid JSON in following format, and AVOID QUESTION SPECIFIC WORDING in reasoning_path keys:
{
  "route": {
    "category": "<High level type of question>",
    "intent": ["<Goal 1>", "<Goal 2>"],
    "difficulty": <1|2|3>,
    "budget": <Follow Budget Contract>,
    "reasoning_path": ["<DescriptiveStepName1>", "<DescriptiveStepName2>", ...]
  },
  "rationale": {
    "<DescriptiveStepName1>": "<short reasoning with derivation>",
    "<DescriptiveStepName2>": "<short reasoning with derivation>",
    ...
  },
  "ans": <numeric>
}
- Category: The high level type of the question.
- Intent: The high level goal of the question.
- Difficulty: How hard is this question (1-3).
- Budget: The maximum number of steps allowed.
"""

Structured_RPB_PART_THREE_DETAILED = """
Output ONLY valid JSON in following format, and AVOID QUESTION SPECIFIC WORDING in reasoning_path keys:
{
  "route": {
    "category": "<High level type of question>",
    "intent": ["<Goal 1>", "<Goal 2>"],
    "difficulty": <1|2|3>,
    "budget": <Follow Budget Contract>,
    "reasoning_path": ["<DescriptiveStepName1>", "<DescriptiveStepName2>", ...]
  },
  "rationale": {
    "<DescriptiveStepName1>": "Step1: <...> Step2: <...> Step3: <...>",
    "<DescriptiveStepName2>": "Step1: <...> Step2: <...>",
    ...
  },
  "ans": <numeric>
}
- Category: The high level type of the question.
- Intent: The high level goal of the question.
- Difficulty: How hard is this question (1-3).
- Budget: The maximum number of steps allowed.
- Rationale: For each reasoning_path entry, write a detailed multi-step explanation in a single string.
  Use Step1:, Step2:, Step3:, ... labels; choose as many substeps as needed (they do NOT count toward budget).
"""
def wrap_structured_category_based_freeform_prompt(
    question,
    category="",
    intents=None,
    answer=None,
    task_type="math",
    detailed: bool = False,
):
    assert task_type in {"math", "text"}, "task_type must be 'math' or 'text'"
    category_text = str(category or "").strip()
    if isinstance(intents, (list, tuple)):
        intents_text = ", ".join(str(item).strip() for item in intents if str(item).strip())
    else:
        intents_text = str(intents or "").strip()

    if task_type == "math":
        task_instruction = """You are solving a math or word problem.
    - Decompose the problem into abstract reasoning steps.
    - Reasoning each step with a general action name (no question-specific words).
    - Each step must show actual computation."""
    else:
        task_instruction = """You are solving a commonsense, causal, or logical reasoning task.
    - Decompose the reasoning into abstract steps (e.g., RetrieveFact, InferConsequence).
    - Each step should express the reasoning explicitly and reflects inference or logic."""

    rationale_requirement = "2. Each StepReasoning field must contain short, essential reasoning"
    detailed_requirements = ""
    if detailed:
        rationale_requirement = (
            "2. Each StepReasoning field must contain detailed, stepwise reasoning with derivations"
        )
        detailed_requirements = """

    Detailed rationale requirements:
    - Each StepReasoning field must be a single string with labeled substeps (Step1:, Step2:, ...).
    - Use as many substeps as needed for clarity; keep each substep concise and computational.
    """

    teacher_requirements = f"""
    Hard rules:
    - Output must match the required JSON schema exactly.
    - "route" must be a dictionary containing exactly: difficulty, budget, and reasoning_path.
    - "reasoning_path" must be a list of strings representing the ordered steps.
    - The keys in "rationale" must match the strings in "reasoning_path" exactly.

    Budget contract:
    - Difficulty 1 implies Budget 2.
    - Difficulty 2 or 3 implies Budget 3.

    Field-count contract:
    - The length of the "reasoning_path" list must be NO MORE THAN the <budget>.

    Reasoning key naming policy (CRITICAL):
    1. Format: Keys (in reasoning_path and rationale) must be TitleCase letters only (A-Z, a-z). No spaces, underscores, digits.
    2. Semantics: Keys MUST be high-level descriptive summaries of the action taken in that step, avoid question specific wording.
    3. FORBIDDEN: Do NOT use generic sequential names (e.g., StepOne, Calculation1).

    Rationale requirements:
    1. StepReasoning field names must be human-readable
    {rationale_requirement}
    {detailed_requirements}
    """

    answer_hint = f"\nThe correct answer is: {answer}\n" if answer is not None else ""
    category_line = f"\nCategory: {category_text}" if category_text else ""
    intent_line = f"\nIntents: {intents_text}" if intents_text else ""

    prompt_body = (
        Structured_CATEGORY_BASED_FREEFORM_PART_THREE_DETAILED
        if detailed
        else Structured_CATEGORY_BASED_FREEFORM_PART_THREE
    )
    if task_type == "text":
        prompt_body = prompt_body.replace("<numeric>", "<bool>")

    return (
        f"{task_instruction}\n\n"
        f"Question: {question}{category_line}{intent_line}{answer_hint}\n"
        f"{prompt_body.strip()}\n"
        f"{teacher_requirements}"
    )


def wrap_structured_freeform_prompt(question, answer=None, task_type="math", detailed: bool = False):
    assert task_type in {"math", "text"}, "task_type must be 'math' or 'text'"

    structured_freeform_three = """
    Output ONLY valid JSON in this format, and AVOID QUESTION SPECIFIC WORDING in reasoning_path keys:
    {
      "route": {
        "category": "<High level type of question>",
        "intent": ["<Goal 1>", "<Goal 2>"],
        "difficulty": <1|2|3>,
        "budget": <Follow Budget Contract>,
        "reasoning_path": ["<DescriptiveStepName1>", "<DescriptiveStepName2>", ...]
      },
      "rationale": {
        "<DescriptiveStepName1>": "<short reasoning with derivation>",
        "<DescriptiveStepName2>": "<short reasoning with derivation>",
        ...
      },
      "ans": <numeric>
    }
    - Category: The high level type of the question.
    - Intent: The high level goal of the question.
    - Difficulty: How hard is this question (1-3).
    - Budget: The maximum number of steps allowed.
    """
    if detailed:
        structured_freeform_three = Structured_RPB_PART_THREE_DETAILED
    if task_type == "text":
        structured_freeform_three = structured_freeform_three.replace("<numeric>", "<bool>")

    if task_type == "math":
        task_instruction = """You are solving a math or word problem.
    - Decompose the problem into abstract reasoning steps.
    - Reasoning each step with a general action name (no question-specific words).
    - Each step must show actual computation."""

    else:
        task_instruction = """You are solving a commonsense, causal, or logical reasoning task.
    - Decompose the reasoning into abstract steps (e.g., RetrieveFact, InferConsequence).
    - Each step should express the reasoning explicitly and reflects inference or logic."""


    rationale_requirement = "2. Each StepReasoning field must contain short, essential reasoning"
    detailed_requirements = ""
    if detailed:
        rationale_requirement = (
            "2. Each StepReasoning field must contain detailed, stepwise reasoning with derivations"
        )
        detailed_requirements = """

    Detailed rationale requirements:
    - Each StepReasoning field must be a single string with labeled substeps (Step1:, Step2:, ...).
    - Use as many substeps as needed for clarity; keep each substep concise and computational.
    """

    teacher_requirements = f"""
    Hard rules:
    - Output must match the required JSON schema exactly.
    - "route" must be a dictionary containing exactly: category, intent, difficulty, budget, and reasoning_path.
    - "reasoning_path" must be a list of strings representing the ordered steps.
    - The keys in "rationale" must match the strings in "reasoning_path" exactly.

    Budget contract:
    - Difficulty 1 implies Budget 2.
    - Difficulty 2 or 3 implies Budget 3.

    Field-count contract:
    - The length of the "reasoning_path" list must be NO MORE THAN the <budget>.

    Reasoning key naming policy (CRITICAL):
    1. Format: Keys (in reasoning_path and rationale) must be TitleCase letters only (A-Z, a-z). No spaces, underscores, digits.
    2. Semantics: Keys MUST be high-level descriptive summaries of the action taken in that step, avoid question specific wording.
    3. FORBIDDEN: Do NOT use generic sequential names (e.g., StepOne, Calculation1).

    Rationale requirements:
    1. StepReasoning field names must be human-readable
    {rationale_requirement}
   
    {detailed_requirements}
    """

    answer_hint = f"\nThe correct answer is: {answer}\n" if answer is not None else ""

    return f"{task_instruction}\n\nQuestion: {question}{answer_hint}\n{structured_freeform_three}\n{teacher_requirements}"


FREEFORM_REASONING_PATH_PART_THREE = """
Output ONLY valid JSON with freely chosen reasoning paths:
{
  "rationale": {
    "<DescriptiveStepName1>": "<concise derivation>",
    "<DescriptiveStepName2>": "<concise derivation>",
    ...
  },
  "ans": <numeric>
}

Rules:
- Freely choose up to three reasoning paths with high-level, TitleCase names (letters only; no spaces, digits, or underscores).
- Keys must be human-readable action labels (e.g., PlanComputation, CombineTotals); avoid generic names like ReasoningPath1 and avoid question-specific wording.
- Keep each rationale concise and focused on the computation or logic.
- If a second path is unnecessary, omit it.
- "ans" must be numeric (no strings, units, or words).
- Do not include any markdown or extra text; return JSON only.
"""

FREEFORM_REASONING_PATH_PART_THREE_DETAILED = """
Output ONLY valid JSON with freely chosen reasoning paths:
{
  "rationale": {
    "<DescriptiveStepName1>": "Step1: <...> Step2: <...>",
    "<DescriptiveStepName2>": "Step1: <...> Step2: <...>",
    ...
  },
  "ans": <numeric>
}

Rules:
- Freely choose up to three reasoning paths with high-level, TitleCase names (letters only; no spaces, digits, or underscores).
- Keys must be human-readable action labels (e.g., PlanComputation, CombineTotals); avoid generic names like ReasoningPath1 and avoid question-specific wording.
- Each rationale must be a single string with labeled substeps (Step1:, Step2:, ...).
- Keep substeps explicit about calculations or logical moves.
- If a second path is unnecessary, omit it.
- "ans" must be numeric (no strings, units, or words).
- Do not include any markdown or extra text; return JSON only.
"""


def wrap_freeform_prompt(
    question: str, answer: str | None = None, task_type: str = "math", detailed: bool = False
) -> str:
    """Freeform reasoning without a predefined path bank."""
    prompt_body = (
        FREEFORM_REASONING_PATH_PART_THREE_DETAILED
        if detailed
        else FREEFORM_REASONING_PATH_PART_THREE
    )
    text_task_types = {"text", "commonsense", "common", "strategyqa", "strategy_qa"}
    if str(task_type).lower() in text_task_types:
        prompt_body = (
            prompt_body.replace("<numeric>", "<bool>")
            .replace('"ans": <numeric>', '"ans": <bool>')
            .replace('"ans" must be numeric (no strings, units, or words).', '"ans" must be true or false.')
            .replace('- "ans" must be numeric (no strings, units, or words).', '- "ans" must be true or false.')
        )
    answer_hint = f"\nAnswer hint: {answer}" if answer is not None else ""
    return f"\nQuestion: {question}{answer_hint}\n\n{prompt_body.strip()}"


SECOND_ROUND_MATH_REASONING_SYSTEM_PROMPT = """
You are a structured reasoning tutor.
You will be given:
- A math question
- One or more routing plans including:
    - Category
    - Intent
    - Budget
    - ReasoningPath (ordered reasoning step names)
Pick the best routing plan from options and follow it if it is adequate; otherwise, refine only the reasoning_path conservatively.
"""
SECOND_ROUND_MATH_REASONING_PART_THREE_PROMPT = """
Task Instructions:

1) Assess reasoning_path suitability:
   - If the provided reasoning_path can solve the question, keep it.
   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.
   - Keep category, intent, difficulty, and budget unchanged.

2) Generate rationale:
   - For each reasoning_path in the chosen reasoning_path, create a concise reasoning step that include derivation.
   - The number of rationale steps must not exceed budget.

3) Final answer:
   - Compute the numeric answer and place it in "ans".

Output (strict JSON, no markdown or extra text):
{
  "route": { ...same as input, but reasoning_path updated if revised... },
  "rationale": {
    "ReasoningPath1": "<...derivation...>",
    "ReasoningPath2": "<...derivation...>"
  },
  "ans": <numeric>
}

Requirements:
- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.
- Keep reasoning_path names exactly as used in reasoning_path.
- Do not exceed budget in number of steps.
"""

SECOND_ROUND_MATH_REASONING_PART_THREE_OPTIONS_PROMPT = """
Task Instructions:

1) Assess reasoning_path suitability:
   - If the provided reasoning_path can solve the question, keep it.
   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.
   - Keep category, intent, difficulty, and budget unchanged.

2) Generate rationale:
   - For each reasoning_path in the chosen reasoning_path, create a concise reasoning step that include derivation.
   - The number of rationale steps must not exceed budget.

3) Final answer:
   - Compute the numeric answer and place it in "ans".

Output (strict JSON, no markdown or extra text):
{
  "route": { ...same as input, but reasoning_path updated if revised... },
  "rationale": {
    "ReasoningPath1": "<...derivation...>",
    "ReasoningPath2": "<...derivation...>"
  },
  "ans": <numeric>
}

Requirements:
- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.
- Keep reasoning_path names exactly as used in reasoning_path.
- Do not exceed budget in number of steps.
"""

SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_PROMPT = """
Task Instructions:

1) Assess reasoning_path suitability:
   - If the provided reasoning_path can solve the question, keep it.
   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.
   - Keep category, intent, difficulty, and budget unchanged.

2) Generate detailed rationale:
   - For each reasoning_path in the chosen reasoning_path, create a detailed rationale string labeled Step1:, Step2:, Step3:, ...
   - Use as many substeps as needed; substeps do NOT count toward budget.
   - Each substep should include explicit derivation or computation where applicable.

3) Final answer:
   - Compute the numeric answer and place it in "ans".

Output (strict JSON, no markdown or extra text):
{
  "route": { ...same as input, but reasoning_path updated if revised... },
  "rationale": {
    "ReasoningPath1": "Step1: <...> Step2: <...>",
    "ReasoningPath2": "Step1: <...> Step2: <...>"
  },
  "ans": <numeric>
}

Requirements:
- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.
- Keep reasoning_path names exactly as used in reasoning_path.
- Do not exceed budget in number of reasoning_path steps.
- Rationale values must be single strings with Step1:, Step2:, ... labels.
"""

SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_OPTIONS_PROMPT = """
Task Instructions:

1) Assess reasoning_path suitability:
   - If the provided reasoning_path can solve the question, keep it.
   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.
   - Keep category, intent, difficulty, and budget unchanged.

2) Generate detailed rationale:
   - For each reasoning_path in the chosen reasoning_path, create a detailed rationale string labeled Step1:, Step2:, Step3:, ...
   - Use as many substeps as needed; substeps do NOT count toward budget.
   - Each substep should include explicit derivation or computation where applicable.

3) Final answer:
   - Compute the numeric answer and place it in "ans".

Output (strict JSON, no markdown or extra text):
{
  "route": { ...same as input, but reasoning_path updated if revised... },
  "rationale": {
    "ReasoningPath1": "Step1: <...> Step2: <...>",
    "ReasoningPath2": "Step1: <...> Step2: <...>"
  },
  "ans": <numeric>
}

Requirements:
- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.
- Keep reasoning_path names exactly as used in reasoning_path.
- Do not exceed budget in number of reasoning_path steps.
- Rationale values must be single strings with Step1:, Step2:, ... labels.
"""


def build_second_round_system_prompt(top_k_paths: int = 1) -> str:
    base = SECOND_ROUND_MATH_REASONING_SYSTEM_PROMPT.strip()
    target = "- ReasoningPath (ordered reasoning step names)"
    pick_line = (
        "Pick the best routing plan from options and follow it if it is adequate; otherwise, "
        "refine only the reasoning_path conservatively."
    )
    single_line = (
        "Follow the provided routing plan if it is adequate; otherwise, refine only the "
        "reasoning_path conservatively."
    )
    replacement = (
        "- ReasoningPathOptions (candidate paths)"
        if top_k_paths > 1
        else target
    )
    lines = base.splitlines()
    updated = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped == target:
            updated.append(replacement)
            replaced = True
        elif stripped == pick_line and top_k_paths <= 1:
            updated.append(single_line)
        else:
            updated.append(line)
    if not replaced:
        updated.append(replacement)
    return "\n".join(updated)


def build_second_round_prompt(
    question: str,
    route: dict,
    answer: str | None = None,
    detailed: bool = False,
    top_k_paths: int = 1,
) -> str:
    """Build the complete round-2 prompt (system + question/route + instructions)."""
    ans_hint = f"\nAnswer hint: {answer}" if answer is not None else ""
    use_options = top_k_paths > 1
    if detailed:
        part_three = (
            SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_OPTIONS_PROMPT
            if use_options
            else SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_PROMPT
        )
    else:
        part_three = (
            SECOND_ROUND_MATH_REASONING_PART_THREE_OPTIONS_PROMPT
            if use_options
            else SECOND_ROUND_MATH_REASONING_PART_THREE_PROMPT
        )
    return (
        f"{build_second_round_system_prompt(top_k_paths).strip()}\n\n"
        f"Question: {question}{ans_hint}\n"
        f"Route: {json.dumps(route, ensure_ascii=False)}\n\n"
        f"{part_three.strip()}"
    )


STRUCTURED_FREE_FORM_PART_THREE = """
Reason freely using up to three reasoning_paths under "rationale".
Choose reasoning_path names appropriate to the problem.

   
Output (strict JSON, no markdown or extra text):
{
  "rationale": {
    "StepReasoning1": "<...derivation ...>",
    "StepReasoning2": "<...derivation ...>",
    "StepReasoning3_Optional": "<...derivation ...>"
  },
  "ans": <numeric final answer>
}


Rules:
- For each reasoning_path, create a reasoning step with derivation.
- Use no more than three rationale fields.
- Reasoning names must be human-readable.
- Do not use abbreviations.
- JSON only.
- "ans" must be numeric.
"""
