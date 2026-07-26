"""OpportunityValidator — integrity checks on discovered opportunities."""

from __future__ import annotations

from pydantic import BaseModel, Field

from phase3.opportunity.schema import Opportunity, OpportunityMetadata


class ValidationResult(BaseModel):
    """Result of opportunity validation."""

    model_config = {"frozen": True, "extra": "forbid"}

    valid: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    opportunities_checked: int = 0
    duplicate_count: int = 0
    missing_evidence_count: int = 0
    broken_reference_count: int = 0
    invalid_score_count: int = 0
    missing_confidence_count: int = 0
    schema_mismatch_count: int = 0


class OpportunityValidator:
    """Validates opportunity data integrity."""

    def validate(
        self,
        opportunities: list[Opportunity],
        metadata: OpportunityMetadata | None = None,
        valid_chain_ids: set[str] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        opp_count = len(opportunities)

        if opp_count == 0:
            warnings.append("No opportunities to validate")
            return ValidationResult(
                valid=True,
                warnings=warnings,
                opportunities_checked=0,
            )

        # 1. Duplicate opportunity IDs
        ids = [o.opportunity_id for o in opportunities]
        dupes = set(i for i in ids if ids.count(i) > 1)
        if dupes:
            errors.append(f"Duplicate opportunity IDs: {len(dupes)}")
            duplicate_count = len(dupes)
        else:
            duplicate_count = 0

        # 2. Missing evidence
        missing_evidence = sum(1 for o in opportunities if len(o.supporting_evidence) == 0)
        if missing_evidence:
            warnings.append(f"{missing_evidence} opportunities have no supporting evidence")

        # 3. Broken reasoning references
        broken_refs = 0
        if valid_chain_ids:
            for o in opportunities:
                for rid in o.reasoning_chain_ids:
                    if rid not in valid_chain_ids:
                        broken_refs += 1
            if broken_refs:
                warnings.append(f"{broken_refs} broken reasoning chain references")

        # 4. Invalid scores
        invalid_scores = 0
        for o in opportunities:
            if not (0.0 <= o.opportunity_score <= 1.0):
                invalid_scores += 1
        if invalid_scores:
            errors.append(f"{invalid_scores} opportunities have out-of-range scores")

        # 5. Missing confidence
        missing_conf = sum(
            1 for o in opportunities if o.confidence.final_confidence == 0.0 and o.opportunity_score > 0
        )
        if missing_conf:
            warnings.append(f"{missing_conf} opportunities have zero confidence despite non-zero score")

        # 6. Schema consistency
        schema_mismatch = 0
        for o in opportunities:
            if o.pipeline_version != "1.0" or o.schema_version != "1.0":
                schema_mismatch += 1

        # 7. Manifest consistency
        if metadata is not None:
            if metadata.total_opportunities != opp_count:
                warnings.append(
                    f"Manifest reports {metadata.total_opportunities} opportunities, "
                    f"but {opp_count} provided"
                )

        valid = len(errors) == 0

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            opportunities_checked=opp_count,
            duplicate_count=duplicate_count,
            missing_evidence_count=missing_evidence,
            broken_reference_count=broken_refs,
            invalid_score_count=invalid_scores,
            missing_confidence_count=missing_conf,
            schema_mismatch_count=schema_mismatch,
        )
