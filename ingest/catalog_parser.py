import json
from pathlib import Path

from ingest.models import Control


class CatalogParser:
    def __init__(self, include_withdrawn: bool = False, include_enhancements: bool = True):
        self.include_withdrawn = include_withdrawn
        self.include_enhancements = include_enhancements

    def parse(self, catalog_path: Path) -> list[Control]:
        with open(catalog_path, encoding="utf-8") as f:
            data = json.load(f)

        controls = []
        for group in data["catalog"]["groups"]:
            controls.extend(
                self._parse_control_tree(group["controls"], group["id"], group["title"])
            )
        return controls

    def _parse_control_tree(
        self, control_jsons: list[dict], family_id: str, family_title: str
    ) -> list[Control]:
        """Parse a list of controls, recursing into nested enhancement controls."""
        controls = []
        for control_json in control_jsons:
            if not self.include_withdrawn and self._is_withdrawn(control_json):
                continue
            controls.append(self._parse_control(control_json, family_id, family_title))

            enhancements = control_json.get("controls", [])
            if enhancements and self.include_enhancements:
                controls.extend(
                    self._parse_control_tree(enhancements, family_id, family_title)
                )
        return controls

    def _parse_control(self, control_json: dict, family_id: str, family_title: str) -> Control:
        return Control(
            id=control_json["id"],
            family_id=family_id,
            family_title=family_title,
            title=control_json["title"],
            statement=self._extract_statement(control_json),
            guidance=self._extract_guidance(control_json),
            assessment_objectives=self._extract_assessment_objectives(control_json),
            is_withdrawn=self._is_withdrawn(control_json),
        )

    def _is_withdrawn(self, control_json: dict) -> bool:
        return any(
            prop.get("name") == "status" and prop.get("value") == "withdrawn"
            for prop in control_json.get("props", [])
        )

    def _extract_statement(self, control_json: dict) -> str:
        # Some controls (e.g. pe-13) put prose directly on the statement part;
        # others (e.g. ac-1) break it into lettered sub-items with no prose on
        # the parent. _collect_prose handles both shapes.
        statement_part = self._find_part(control_json, "statement")
        if statement_part is None:
            return ""
        return "\n".join(self._collect_prose(statement_part))

    def _extract_guidance(self, control_json: dict) -> str:
        guidance_part = self._find_part(control_json, "guidance")
        if guidance_part is None:
            return ""
        return guidance_part.get("prose", "")

    def _extract_assessment_objectives(self, control_json: dict) -> list[str]:
        objective_part = self._find_part(control_json, "assessment-objective")
        if objective_part is None:
            return []
        return self._collect_prose(objective_part)

    def _collect_prose(self, part: dict) -> list[str]:
        """Assessment objectives nest to arbitrary depth (e.g. ac-1_obj.a-1, .a-2);
        only leaf parts carry 'prose', so walk the whole subtree to collect them."""
        prose = [part["prose"]] if "prose" in part else []
        for sub_part in part.get("parts", []):
            prose.extend(self._collect_prose(sub_part))
        return prose

    def _find_part(self, control_json: dict, part_name: str) -> dict | None:
        for part in control_json.get("parts", []):
            if part.get("name") == part_name:
                return part
        return None
