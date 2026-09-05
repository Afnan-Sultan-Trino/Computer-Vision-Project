"""
DAY 5: Lightweight rule-based "ontology" layer.

For the poster demo, a full formal OWL/Turtle ontology is not necessary —
a clean, well-documented rule table (structured as a Python dict / simple
graph) proves the concept and can be upgraded to real OWL later using
owlready2 if you continue toward a journal version.

Structure: lesion_pattern -> body_location -> clinical explanation +
exposure-risk note. Rules are illustrative; refine wording with a
clinician/literature before using in the final poster or paper.
"""

from dataclasses import dataclass


@dataclass
class OntologyRule:
    lesion_pattern: str
    body_location: str
    explanation: str
    risk_note: str


# ---- Minimal rule base (expand to 10-15 rules for the poster) ----
RULES = [
    OntologyRule(
        lesion_pattern="diffuse_hyperpigmentation",
        body_location="trunk",
        explanation=(
            "Diffuse spotted hyperpigmentation (melanosis) on the trunk is "
            "one of the earliest cutaneous signs of chronic arsenic exposure."
        ),
        risk_note=(
            "Associated with long-term ingestion of contaminated groundwater; "
            "recommend confirmatory water-source testing and dermatology referral."
        ),
    ),
    OntologyRule(
        lesion_pattern="keratotic_lesion",
        body_location="palm",
        explanation=(
            "Punctate or diffuse keratosis on the palm is a classical marker of "
            "arsenical keratosis, often appearing after years of exposure."
        ),
        risk_note=(
            "Palmar/plantar keratosis carries elevated risk of progression to "
            "Bowen's disease if untreated; recommend closer monitoring."
        ),
    ),
    OntologyRule(
        lesion_pattern="keratotic_lesion",
        body_location="sole",
        explanation=(
            "Keratotic thickening on the sole is consistent with arsenical "
            "keratosis, commonly co-occurring with palmar lesions."
        ),
        risk_note="Same progression risk as palmar keratosis; monitor both sites together.",
    ),
    OntologyRule(
        lesion_pattern="leucomelanosis",
        body_location="limbs",
        explanation=(
            "Mixed hypo/hyperpigmented ('rain-drop') patterning on the limbs "
            "is a recognized intermediate-stage arsenicosis presentation."
        ),
        risk_note="Suggests moderate-duration exposure; recommend follow-up imaging in 6 months.",
    ),
    # ... add 6-10 more rules covering other lesion/location combinations
]


def match_rule(lesion_pattern: str, body_location: str) -> OntologyRule | None:
    for rule in RULES:
        if rule.lesion_pattern == lesion_pattern and rule.body_location == body_location:
            return rule
    return None


def generate_report(cnn_label: str, cnn_confidence: float,
                     lesion_pattern: str, body_location: str) -> str:
    """
    Combines CNN output with the ontology rule to build a human-readable
    screening note instead of a raw probability. This function is what
    04_faithfulness_check.py and 05_demo_app.py both call.
    """
    if cnn_label != "Arsenic-affected":
        return (
            f"Prediction: {cnn_label} (confidence {cnn_confidence:.1%}). "
            "No arsenic-related lesion pattern detected in this image."
        )

    rule = match_rule(lesion_pattern, body_location)
    if rule is None:
        return (
            f"Prediction: Arsenic-affected (confidence {cnn_confidence:.1%}). "
            f"Lesion pattern '{lesion_pattern}' at '{body_location}' is not yet "
            "covered by the rule base — flag for manual dermatologist review."
        )

    return (
        f"Prediction: Arsenic-affected (confidence {cnn_confidence:.1%}).\n"
        f"Clinical reasoning: {rule.explanation}\n"
        f"Risk note: {rule.risk_note}"
    )


# ---- Quick manual test ----
if __name__ == "__main__":
    # In the real pipeline, lesion_pattern/body_location would come from a
    # small classifier or manual tagging on your labeled subset — for the
    # poster demo you can hardcode a handful of examples like this:
    report = generate_report(
        cnn_label="Arsenic-affected",
        cnn_confidence=0.94,
        lesion_pattern="keratotic_lesion",
        body_location="palm",
    )
    print(report)
