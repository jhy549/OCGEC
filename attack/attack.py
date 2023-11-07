from abc import ABC, abstractclassmethod

class Attack(ABC):

    def __init__(self):
        self.name = "attack"
    
    @abstractclassmethod
    def attack(self):
        ...

class InputModifyAttack(Attack):

    def __init__(self):
        super().__init__()
        self.name = "InputModifyAttack"

class ModelRetrainAttack(Attack):

    def __init__(self) -> None:
        super().__init__()
        self.name = "ModelRetrainAttack"

class BenignAttack(InputModifyAttack):

    def __init__(self):
        super().__init__()
        self.name = "BadNet"
    
    def attack(self, input):
        return input 
    