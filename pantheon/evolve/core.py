import random
import uuid

from pydantic import BaseModel, Field

from ..utils.log import logger
from ..agent import Agent


class Program(BaseModel):
    """Represents a program in the evolution process."""
    id: str = Field(description="The id of the program", default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(description="The description of the program", default="")
    code: str = Field(description="The code of the program", default="")
    score: float = Field(description="The score of the program, higher is better", default=0.0)
    parents: list[str] = Field(description="The ids of the parent programs", default_factory=list)


class ProgramDB:
    def __init__(self):
        self.generations: list[list[Program]] = []

    def add_program(self, generation: int, program: Program):
        if len(self.generations) <= generation:
            self.generations.append([])
        self.generations[generation].append(program)

    def get_program(self, id: str, generation: int | None = None) -> Program | None:
        if generation is None:
            for gen in self.generations:
                for program in gen:
                    if program.id == id:
                        return program
        else:
            for program in self.generations[generation]:
                if program.id == id:
                    return program
        return None

    def select_variants(self, num: int) -> list[Program]:
        latest_generation = self.generations[-1]
        
        if not latest_generation:
            return []
        
        # Handle edge cases
        if num <= 0:
            return []
        if num >= len(latest_generation):
            return latest_generation.copy()
        
        # Calculate weights based on scores
        # Add a small epsilon to avoid zero weights
        epsilon = 1e-10
        min_score = min(p.score for p in latest_generation)
        
        # Shift scores to ensure all weights are positive
        # If all scores are negative, shift them up
        if min_score < 0:
            weights = [p.score - min_score + epsilon for p in latest_generation]
        else:
            weights = [max(p.score, epsilon) for p in latest_generation]
        
        # Use random.choices for weighted selection without replacement
        selected = random.choices(
            latest_generation,
            weights=weights,
            k=num
        )
        
        return selected


class Evolve:
    def __init__(
        self,
        agent: Agent,
        init_code: str | None = None,
        program_db: ProgramDB | None = None,
        max_generations: int = 10,
        varients_per_generation: int = 10,
        max_parents: int = 2,
    ):
        self.agent = agent
        if init_code is not None:
            program_db = ProgramDB()
            program_db.add_program(0, Program(code=init_code))
        else:
            assert program_db is not None, "program_db is required when init_code is not provided"
            assert len(program_db.generations) == 0, "program_db must be empty when init_code is not provided"
        self.program_db = program_db
        self.max_generations = max_generations
        self.varients_per_generation = varients_per_generation

    def run(self):
        generation = 0
        num_varients = self.varients_per_generation
        while generation < self.max_generations:
            pass
