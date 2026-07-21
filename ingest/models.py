from dataclasses import dataclass,field

@dataclass
class Control:
    id:str
    family_id:str
    family_title:str
    title:str
    statement:str
    guidance:str
    assessment_objectives:list[str]=field(default_factory=list)
    is_withdrawn:bool=False


    def to_chunk_text(self) -> str:
        lines = [f"{self.id.upper()} — {self.title}", "", self.statement]
        if self.guidance:
            lines += ["", "Guidance:", self.guidance]
        if self.assessment_objectives:
            lines += ["", "Assessment objectives:"]
            lines += [f"- {objective}" for objective in self.assessment_objectives]
        return "\n".join(lines)
