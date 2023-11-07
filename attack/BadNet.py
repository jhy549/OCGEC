from .attack import InputModifyAttack


class BadNetAttack(InputModifyAttack):

    def __init__(self):
        super().__init__()
        self.name = "BadNet"
    
    def attack(self, input):
        return input 
