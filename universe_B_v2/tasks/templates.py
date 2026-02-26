"""
tasks/templates.py
─────────────────────────────────────────────────────────────
Ten subjective task template families, each capable of generating
N independent prompt variations via seeded randomisation.

Design rules
────────────
• NO objective ground truth — all tasks are interpretive/argumentative.
• Each variation is fully self-contained (no cross-iteration state).
• Prompts are compact but rich enough to elicit multi-paragraph responses.
• Variation seeds derive from iteration_id to guarantee reproducibility
  while avoiding identical prompts across iterations.
• The TaskRegistry maps template_id (0–9) to a generator callable.

Template families (10)
──────────────────────
 0  policy_analysis          — Evaluate a hypothetical policy proposal
 1  strategic_decision       — Choose between strategic options under uncertainty
 2  ethical_dilemma          — Navigate a multi-stakeholder ethical conflict
 3  argument_construction    — Build a structured argument for/against a position
 4  design_critique          — Critique a described system/service design
 5  risk_assessment          — Assess and prioritise risks for a described scenario
 6  negotiation_strategy     — Design a negotiation strategy in a described standoff
 7  system_design            — Propose a design for a described sociotechnical system
 8  comparative_evaluation   — Compare and rank two described approaches
 9  creative_essay           — Write a structured analytical essay on an open theme
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskVariation:
    template_id: int
    variation_id: int
    family_name: str
    prompt: str
    # Seed used to generate this variation (for audit trail)
    variation_seed: int


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary pools for seeded variation
# ──────────────────────────────────────────────────────────────────────────────

_POLICY_DOMAINS = [
    "universal basic income in post-automation labour markets",
    "mandatory digital literacy education in secondary schools",
    "carbon taxation on consumer goods rather than producers",
    "publicly funded mental health infrastructure in rural regions",
    "four-day working weeks mandated by national legislation",
    "algorithmic hiring audits required by employment law",
    "urban car-free zones in cities with over one million residents",
    "mandatory GMO labelling on all processed food products",
    "national data sovereignty laws restricting cross-border data flows",
    "decriminalisation of drug possession for personal use",
    "mandatory parental leave for both primary and secondary caregivers",
    "a sovereign wealth fund seeded by windfall taxes on fossil fuel profits",
    "universal pre-school education funded by a payroll levy",
    "public ownership of broadband infrastructure in rural areas",
    "ranked-choice voting in national legislative elections",
    "government-backed starter home schemes for first-time buyers under 35",
    "sugar taxes extended to all ultra-processed foods",
    "free public transit in metropolitan areas on weekends",
    "mandatory climate impact disclosures for publicly traded companies",
    "a guaranteed minimum internet speed as a public utility baseline",
]

_STRATEGIC_CONTEXTS = [
    "a mid-sized software startup deciding between rapid geographic expansion and deep product consolidation",
    "a regional hospital system choosing between centralising specialist care or distributing outpatient services",
    "a non-profit educational publisher choosing between open-access and subscription revenue models",
    "a manufacturing firm weighing near-shoring supply chains versus maintaining lower-cost offshore operations",
    "a national museum deciding between digital collection archiving and increasing physical accessibility",
    "a city government choosing between public EV charging infrastructure or incentivising private deployment",
    "a research university debating whether to merge its AI and cognitive science departments",
    "a fintech startup choosing between regulatory arbitrage in a permissive jurisdiction or early compliance investment",
    "a food cooperative deciding between vertical integration of supply or deepening community ownership",
    "a public broadcaster weighing subscription streaming against licence-fee-funded open broadcasting",
    "a professional sports franchise choosing between short-term championship spending and long-term youth development",
    "a national railway operator debating high-speed corridor investment versus regional network upgrades",
    "a healthcare AI company choosing between selling point solutions or building an integrated platform",
    "a foundation choosing between concentrated bets on a few programmes versus broad portfolio funding",
    "a media company deciding between AI-assisted journalism tools and a human-only editorial model",
]

_ETHICAL_SCENARIOS = [
    "an autonomous vehicle manufacturer must decide how to encode collision trade-off priorities",
    "a genomics company discovers a user's incidental genetic disease risk without consent provisions set",
    "a city surveillance contractor is asked to share real-time footage with immigration enforcement",
    "a hospital AI triage tool systematically under-scores patients from low-income postcode areas",
    "a social media platform must decide whether to remove historically significant but harmful archival content",
    "a pharmaceutical company possesses unpublished trial data suggesting a competitor's drug is hazardous",
    "a remote work platform's keystroke monitoring identifies an employee's mental health deterioration",
    "a water utility discovers contamination below legal thresholds but above WHO advisory levels",
    "a journalist obtains leaked documents via a source who may face prosecution under state secrecy law",
    "a university must decide whether to accept a major donation from a source with a contested human rights record",
    "a climate model team discovers a funder is selectively publishing only subset of their results",
    "an employer's predictive retention model flags a protected class at disproportionate rates",
    "a public health authority must decide whether to mandate vaccines that have very low but non-zero serious adverse rates",
    "a deepfake detection company is asked to license its technology to a government with a poor press freedom record",
    "a bank's loan algorithm denies credit at higher rates in majority-minority neighbourhoods at statistically equal risk levels",
]

_ARGUMENT_TOPICS = [
    "long-term artificial general intelligence development should be governed by an international treaty body",
    "social media companies should bear strict liability for algorithmic content amplification harms",
    "meritocracy as a social organising principle causes more harm than it prevents",
    "the peer review system in academic publishing is no longer fit for purpose",
    "remote work has fundamentally weakened institutional knowledge transfer",
    "gig economy labour classification laws are insufficiently protective in high-income countries",
    "digital advertising revenue models are structurally incompatible with quality journalism",
    "the doctorate as currently structured is an inefficient mechanism for producing researchers",
    "universal surveillance architecture should be prohibited by constitutional amendment",
    "performance-based teacher pay schemes do more harm than good in under-resourced schools",
    "open-source large language models pose greater societal risk than proprietary ones",
    "elected officials should be subject to competency assessments before assuming office",
    "platform monopolies in digital infrastructure are natural and should be regulated rather than broken up",
    "space colonisation should be deprioritised in favour of addressing terrestrial existential risks",
    "the concept of net-zero is a strategic fiction that delays genuine emissions reduction",
]

_DESIGN_SUBJECTS = [
    "a national AI incident reporting registry for high-risk automated decisions",
    "a federated identity layer allowing citizens to authenticate across all government digital services",
    "a peer-support platform for post-incarceration rehabilitative community building",
    "an open data marketplace for municipal sensor networks",
    "a cross-border scientific collaboration grant management system",
    "a distributed telemedicine platform for remote indigenous communities",
    "a collective bargaining coordination tool for gig economy workers",
    "a public-facing algorithmic audit portal for local government automated decisions",
    "a decentralised content moderation appeals system for social media",
    "a national soil health monitoring and reporting platform for smallholder farmers",
    "a student mental health early-warning system integrated into university infrastructure",
    "a crowdsourced translation and verification platform for crisis communications",
    "an open-source carbon footprint tracking API for consumer-facing applications",
    "a multi-stakeholder deliberation platform for participatory budgeting at city scale",
    "a distributed scientific preprint screening service using volunteer expert review",
]

_RISK_SCENARIOS = [
    "a national election body adopting end-to-end internet voting within 18 months",
    "a major urban hospital migrating its electronic health records to a single cloud vendor",
    "a central bank launching a retail central bank digital currency within two years",
    "a city government mandating all public interactions be conducted through a single mobile app",
    "a national food agency allowing cultivated meat products to market without long-term consumption data",
    "a grid operator integrating very high proportions of uncontrolled residential solar without storage mandates",
    "a country eliminating mandatory engineering licencing requirements in the built environment sector",
    "a primary school system replacing classroom teachers with AI tutoring systems as a cost measure",
    "a public transport authority adopting fully driverless autonomous buses on mixed urban roads",
    "a research institution publishing complete synthesis routes for dual-use biochemical compounds openly",
    "a government adopting a single private-sector facial recognition vendor for border control",
    "a regional bank fully automating small-business lending decisions without human review",
    "a city permitting widespread short-term rental conversion of housing stock without occupancy caps",
    "a social network enabling end-to-end encrypted direct payments without transaction monitoring",
    "a university adopting AI-generated examination questions across all undergraduate programmes",
]

_NEGOTIATION_STANDOFFS = [
    "a public sector union and a government in dispute over a hybrid-work reversion mandate",
    "two neighbouring countries in a river basin water allocation dispute during a prolonged drought",
    "a startup and a corporate acquirer with irreconcilable valuations and earnout disagreements",
    "a landlord coalition and a tenant advocacy group over rent stabilisation legislation",
    "competing airline alliances negotiating shared terminal infrastructure at a congested hub airport",
    "a pharmaceutical company and a low-income government over compulsory licensing of a patented vaccine",
    "indigenous land title holders and a renewable energy developer over a wind farm rights-of-way",
    "a platform and a coalition of content creators over algorithmic revenue share transparency",
    "a union of port workers and an automated logistics company over job displacement retraining terms",
    "two academic institutions in a joint graduate programme dispute over intellectual property attribution",
    "a city and a real estate developer over affordable housing inclusion ratios on a prime site",
    "competing public broadcaster factions over merger terms following a government consolidation mandate",
    "a global financial regulator and a DeFi protocol association over KYC compliance standards",
    "a hospital network and an insurer over reimbursement rates for a new class of precision medicine therapies",
    "a national government and a humanitarian NGO over information-sharing arrangements in an active crisis zone",
]

_SYSTEM_THEMES = [
    "a national system for algorithmic accountability in judicial sentencing recommendations",
    "a post-secondary credential verification layer interoperable across 50 participating countries",
    "a participatory science funding allocation system using citizen juries",
    "a decentralised grid balancing market for peer-to-peer renewable energy trading",
    "a structured deliberation system for national digital rights legislation",
    "a cross-agency data-sharing architecture for joined-up homelessness prevention services",
    "a public procurement transparency system for AI systems purchased by government",
    "a federated clinical trial data infrastructure for rare disease research",
    "a national early warning system for food system disruption events",
    "a co-designed public inquiry process allowing affected communities to set investigation scope",
    "an open standard for personalised carbon budgets integrated with consumer banking apps",
    "a multi-modal mobility-as-a-service platform for rural connectivity",
    "a community-controlled data trust for neighbourhood environmental monitoring",
    "a predictive maintenance coordination system for shared critical national infrastructure",
    "a cross-border crisis information commons for coordinated public communication",
]

_COMPARISON_PAIRS = [
    ("mandatory auditing of AI hiring tools", "industry self-certification schemes"),
    ("performance-linked teacher pay", "universal teacher pay uplift"),
    ("congestion charges for urban vehicle access", "public transit expansion without demand management"),
    ("means-tested social transfer programmes", "universal basic income"),
    ("competency-based credentialing", "time-based degree programmes"),
    ("open-source foundation models for public services", "regulated proprietary model licensing"),
    ("participatory budgeting", "representative municipal finance committees"),
    ("platform data portability mandates", "interoperability standards enforcement"),
    ("mandatory national service", "incentivised voluntary public service"),
    ("centralised national data infrastructure", "federated distributed data governance"),
    ("market-rate carbon offsets", "regulatory carbon caps with no offset allowance"),
    ("AI-assisted judicial sentencing", "fully human judicial panels with algorithmic advisory"),
    ("digital textbook mandates in schools", "open-curriculum physical materials"),
    ("concentrated primary care hubs", "distributed community health worker networks"),
    ("algorithmic content moderation at scale", "human editorial moderation with algorithmic triage"),
]

_ESSAY_THEMES = [
    "the role of institutional trust in the adoption of algorithmic decision-making",
    "whether radical transparency in government is compatible with effective public administration",
    "the epistemic consequences of large-scale social media platform design",
    "why complexity science offers a better framework for policy than rational-actor models",
    "the tension between efficiency and resilience in global supply chain architecture",
    "what the history of electrification teaches about the governance of artificial intelligence",
    "the political economy of open-source software in the age of cloud monopolies",
    "whether the precautionary principle is coherent as a regulatory standard for emerging technologies",
    "the normative implications of treating attention as a scarce resource in platform economics",
    "how the concept of 'technical debt' maps onto analogous phenomena in public institutions",
    "the limits of randomised controlled trials as the dominant paradigm in social policy evaluation",
    "what 'informed consent' means in an era of ambient data collection and behavioural inference",
    "the moral significance of reversibility as a design criterion for sociotechnical systems",
    "why coordination failure rather than malice better explains most large-scale technological harms",
    "the relationship between epistemic humility and effective long-range planning",
]


# ──────────────────────────────────────────────────────────────────────────────
# Generator functions (one per family)
# ──────────────────────────────────────────────────────────────────────────────

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _gen_policy_analysis(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    domain = rng.choice(_POLICY_DOMAINS)
    stance = rng.choice(["critically evaluate", "construct a balanced assessment of", "analyse the second-order consequences of"])
    horizon = rng.choice(["a five-year", "a ten-year", "a twenty-year"])
    context = rng.choice(["a high-income democratic state", "a middle-income emerging economy", "a small open economy heavily dependent on trade"])
    return (
        f"You are a senior policy analyst advising government.\n\n"
        f"Task: {stance.capitalize()} the following proposed policy: "
        f"**{domain}**.\n\n"
        f"Your analysis should cover:\n"
        f"1. The core rationale and stated objectives of the policy.\n"
        f"2. Likely first-order effects within {horizon} horizon.\n"
        f"3. Potential unintended consequences and distributional trade-offs.\n"
        f"4. Implementation challenges specific to {context}.\n"
        f"5. A clear overall assessment: whether the policy merits adoption, and under what conditions.\n\n"
        f"Write a structured analytical memo of approximately 400–500 words. "
        f"Avoid generic platitudes. Be specific and intellectually rigorous."
    )


def _gen_strategic_decision(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    context = rng.choice(_STRATEGIC_CONTEXTS)
    constraint = rng.choice([
        "resources are constrained and a pivot is difficult once made",
        "the competitive environment is shifting rapidly and reversibility is valued",
        "stakeholder alignment is fragile and consensus is required",
        "regulatory changes are imminent and may reshape the landscape within 12 months",
    ])
    return (
        f"You are a strategic advisor with deep domain expertise.\n\n"
        f"Scenario: {context.capitalize()}.\n\n"
        f"Additional constraint: {constraint}.\n\n"
        f"Produce a structured strategic recommendation that includes:\n"
        f"1. A diagnosis of the core strategic tension in this scenario.\n"
        f"2. A recommended primary course of action with explicit justification.\n"
        f"3. The key assumptions your recommendation depends upon.\n"
        f"4. A clearly articulated fallback position if those assumptions prove incorrect.\n"
        f"5. Three specific leading indicators you would monitor in the first 90 days.\n\n"
        f"Be concrete. Avoid hedging behind 'it depends' without specifying on what and how."
    )


def _gen_ethical_dilemma(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    scenario = rng.choice(_ETHICAL_SCENARIOS)
    framework = rng.choice([
        "a consequentialist lens",
        "a deontological framework",
        "a virtue ethics perspective",
        "a contractarian framework",
    ])
    return (
        f"You are an applied ethicist advising an organisational ethics review board.\n\n"
        f"Scenario: {scenario.capitalize()}.\n\n"
        f"You have been asked to provide a reasoned ethical analysis. Your response must:\n"
        f"1. Identify all materially affected stakeholders and characterise the harm/benefit distribution.\n"
        f"2. Apply {framework} to evaluate the primary ethical question.\n"
        f"3. Articulate the strongest opposing ethical argument to your conclusion.\n"
        f"4. Propose a defensible course of action, acknowledging residual moral costs.\n"
        f"5. Identify any procedural safeguards that would make the chosen action more justifiable.\n\n"
        f"Write approximately 400–500 words. Demonstrate genuine moral reasoning, not policy boilerplate."
    )


def _gen_argument_construction(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    topic = rng.choice(_ARGUMENT_TOPICS)
    position = rng.choice(["in favour of", "against"])
    form = rng.choice([
        "a persuasive argumentative essay",
        "a structured oral argument outline",
        "a formal position paper introduction",
    ])
    return (
        f"You are preparing {form} for a high-level policy forum.\n\n"
        f"Position: Argue {position} the following claim:\n"
        f"**\"{topic.capitalize()}\"**\n\n"
        f"Your argument must include:\n"
        f"1. A clear thesis statement in the opening paragraph.\n"
        f"2. Three substantive supporting arguments, each with evidence or reasoning.\n"
        f"3. Explicit engagement with the strongest counter-argument.\n"
        f"4. A principled rebuttal or concession-with-qualification of that counter-argument.\n"
        f"5. A closing synthesis that restates why your position holds.\n\n"
        f"Approximately 450 words. Prioritise logical rigour and rhetorical clarity over emotional appeal."
    )


def _gen_design_critique(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    subject = rng.choice(_DESIGN_SUBJECTS)
    flaw_type = rng.choice([
        "scalability and governance model",
        "privacy architecture and data minimisation",
        "incentive alignment between stakeholders",
        "failure modes and resilience under adversarial conditions",
        "accessibility and equity of access across user groups",
    ])
    return (
        f"You are a senior systems architect and design reviewer.\n\n"
        f"You have been presented with a preliminary design proposal for: "
        f"**{subject}**.\n\n"
        f"The design has been flagged for potential weaknesses in its **{flaw_type}**.\n\n"
        f"Provide a rigorous design critique covering:\n"
        f"1. A concise characterisation of the design's stated purpose and core architectural choices.\n"
        f"2. A detailed critique of the flagged dimension, with specific failure scenarios.\n"
        f"3. Two additional design risks you identify beyond the flagged dimension.\n"
        f"4. Concrete redesign recommendations for each identified weakness.\n"
        f"5. An honest assessment of the design's strengths that should be preserved.\n\n"
        f"Write approximately 400–500 words. Be technically specific."
    )


def _gen_risk_assessment(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    scenario = rng.choice(_RISK_SCENARIOS)
    lens = rng.choice([
        "technical and operational risks",
        "social and political risks",
        "economic and market risks",
        "legal and regulatory risks",
        "long-term systemic risks",
    ])
    return (
        f"You are a principal risk analyst at a national advisory body.\n\n"
        f"Scenario under assessment: {scenario.capitalize()}.\n\n"
        f"Focus your assessment primarily on **{lens}**.\n\n"
        f"Produce a structured risk assessment that includes:\n"
        f"1. A brief contextualisation of the scenario and why it carries elevated risk.\n"
        f"2. Identification and description of the five most significant risks in your focus area.\n"
        f"3. A risk priority matrix (likelihood × impact) for each identified risk — described in prose.\n"
        f"4. Recommended mitigations for the top two risks by priority.\n"
        f"5. Residual risks that cannot be fully mitigated and must be accepted or re-scoped.\n\n"
        f"Approximately 450 words. Be specific about mechanisms of harm, not just categories."
    )


def _gen_negotiation_strategy(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    standoff = rng.choice(_NEGOTIATION_STANDOFFS)
    role = rng.choice(["the party with weaker structural leverage", "the party with stronger structural leverage", "a neutral mediator"])
    tactic = rng.choice([
        "integrative bargaining",
        "principled interest-based negotiation",
        "package deal construction",
        "sequenced confidence-building measures",
    ])
    return (
        f"You are a master negotiator and conflict resolution specialist.\n\n"
        f"Standoff: {standoff.capitalize()}.\n\n"
        f"You are advising **{role}**. Apply a **{tactic}** approach.\n\n"
        f"Your strategy document must address:\n"
        f"1. An honest diagnosis of each party's interests, positions, and BATNA.\n"
        f"2. The key sources of value to be created or distributed in a resolution.\n"
        f"3. A concrete opening move or proposal with rationale.\n"
        f"4. Two likely resistance points and how to navigate each.\n"
        f"5. Your definition of a minimally acceptable agreement versus an optimal outcome.\n\n"
        f"Write approximately 450 words. Be tactically specific, not generically advisory."
    )


def _gen_system_design(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    theme = rng.choice(_SYSTEM_THEMES)
    constraint = rng.choice([
        "must function across heterogeneous bureaucratic environments with varying digital maturity",
        "must be deployable within 18 months on a constrained public sector budget",
        "must satisfy simultaneous requirements for transparency, privacy, and auditability",
        "must be resilient to adversarial manipulation by well-resourced actors",
        "must operate without a central coordinating authority",
    ])
    return (
        f"You are a principal architect on a high-stakes public interest technology project.\n\n"
        f"Design brief: Propose a system design for **{theme}**.\n\n"
        f"Hard constraint: The system {constraint}.\n\n"
        f"Your proposal must include:\n"
        f"1. A statement of the core design philosophy and primary objectives.\n"
        f"2. The high-level architectural components and their interactions.\n"
        f"3. How the system addresses the specified hard constraint.\n"
        f"4. The governance and accountability model for the system.\n"
        f"5. The top three technical or organisational risks and your mitigations.\n\n"
        f"Approximately 450 words. Provide enough architectural detail to evaluate feasibility."
    )


def _gen_comparative_evaluation(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    pair = rng.choice(_COMPARISON_PAIRS)
    option_a, option_b = pair
    criterion = rng.choice([
        "long-term equity and distributional outcomes",
        "implementation feasibility in resource-constrained environments",
        "resilience to political capture and regulatory arbitrage",
        "innovation incentive effects",
        "social trust and democratic legitimacy",
    ])
    return (
        f"You are a senior policy evaluator producing a comparative assessment brief.\n\n"
        f"Compare the following two approaches:\n"
        f"  Option A: **{option_a.capitalize()}**\n"
        f"  Option B: **{option_b.capitalize()}**\n\n"
        f"Primary evaluation criterion: **{criterion}**.\n\n"
        f"Your comparative brief must:\n"
        f"1. Characterise each option's core mechanism and logic of change.\n"
        f"2. Evaluate each option against the primary criterion with specificity.\n"
        f"3. Identify two additional dimensions where the options diverge significantly.\n"
        f"4. State a clear overall recommendation with conditions and caveats.\n"
        f"5. Propose a hybrid or sequenced approach if it is demonstrably superior to either standalone option.\n\n"
        f"Approximately 450 words. Avoid false balance — take and defend a position."
    )


def _gen_creative_essay(template_id: int, variation_id: int, seed: int) -> str:
    rng = _rng(seed)
    theme = rng.choice(_ESSAY_THEMES)
    mode = rng.choice([
        "an argumentative analytical essay",
        "a speculative but evidence-grounded essay",
        "an essay structured as a progression of linked claims",
    ])
    return (
        f"You are a scholar-practitioner writing for an intellectually sophisticated general audience.\n\n"
        f"Write {mode} on the following theme:\n\n"
        f"**\"{theme.capitalize()}\"**\n\n"
        f"Your essay should:\n"
        f"1. Open with an arresting framing that positions the reader.\n"
        f"2. Develop three to four substantive ideas that build on each other.\n"
        f"3. Engage critically with at least one widely held view you contest.\n"
        f"4. Draw on specific examples, historical or contemporary, to ground abstract claims.\n"
        f"5. Close with a forward-looking synthesis — what this analysis implies.\n\n"
        f"Approximately 450–500 words. Prioritise intellectual depth and clarity of voice."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_GENERATORS: List[Tuple[str, Callable[[int, int, int], str]]] = [
    ("policy_analysis",        _gen_policy_analysis),
    ("strategic_decision",     _gen_strategic_decision),
    ("ethical_dilemma",        _gen_ethical_dilemma),
    ("argument_construction",  _gen_argument_construction),
    ("design_critique",        _gen_design_critique),
    ("risk_assessment",        _gen_risk_assessment),
    ("negotiation_strategy",   _gen_negotiation_strategy),
    ("system_design",          _gen_system_design),
    ("comparative_evaluation", _gen_comparative_evaluation),
    ("creative_essay",         _gen_creative_essay),
]


class TaskRegistry:
    """
    Central registry that maps (template_id, variation_id) → TaskVariation.

    Seed design
    -----------
    variation_seed = global_seed * 1_000_000 + template_id * 1_000 + variation_id
    This guarantees:
    • Same global_seed → identical experiment.
    • Different template or variation → non-colliding seeds.
    • No state between iterations.
    """

    def __init__(self, global_seed: int = 42):
        self.global_seed = global_seed

    def get(self, template_id: int, variation_id: int) -> TaskVariation:
        assert 0 <= template_id < len(_GENERATORS), \
            f"template_id {template_id} out of range [0, {len(_GENERATORS)-1}]"
        family_name, gen_fn = _GENERATORS[template_id]
        variation_seed = self.global_seed * 1_000_000 + template_id * 1_000 + variation_id
        prompt = gen_fn(template_id, variation_id, variation_seed)
        return TaskVariation(
            template_id=template_id,
            variation_id=variation_id,
            family_name=family_name,
            prompt=prompt,
            variation_seed=variation_seed,
        )

    def iter_all(
        self,
        num_templates: int = 10,
        variations_per_template: int = 100,
    ):
        """Yield all TaskVariations in canonical order: template 0→N, variation 0→M."""
        for tid in range(num_templates):
            for vid in range(variations_per_template):
                yield self.get(tid, vid)

    def get_by_iteration_id(
        self,
        iteration_id: int,
        variations_per_template: int = 100,
    ) -> TaskVariation:
        """Convert a flat iteration_id index to (template_id, variation_id)."""
        template_id = iteration_id // variations_per_template
        variation_id = iteration_id % variations_per_template
        return self.get(template_id, variation_id)

    @property
    def num_families(self) -> int:
        return len(_GENERATORS)

    @property
    def family_names(self) -> List[str]:
        return [name for name, _ in _GENERATORS]
