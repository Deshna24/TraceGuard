import random
from src.attacks.templates import get_random_attack, BENIGN_DISTRACTIONS

class Injector:
    def __init__(self, mode="benign", injection_step_target=1):
        self.mode = mode # 'benign', 'resisted', 'hijacked'
        self.injection_step_target = injection_step_target
        self.has_injected = False
        self.attack_type = None
        self.payload = None
        
        if self.mode in ['resisted', 'hijacked']:
            self.attack_type, self.payload = get_random_attack()
        
    def intercept_observation(self, step_idx: int, original_observation: str) -> str:
        """Inject malicious payload into observation without altering the original goal."""
        
        if self.mode == "benign":
            # Sometimes inject benign distraction for hard negatives
            if random.random() < 0.2:
                return f"{original_observation}\n\n[Context: {random.choice(BENIGN_DISTRACTIONS)}]"
            return original_observation
            
        if self.mode in ["resisted", "hijacked"] and not self.has_injected and step_idx >= self.injection_step_target:
            self.has_injected = True
            
            # Format injection
            injected_obs = f"{original_observation}\n\n[UNTRUSTED CONTENT]\n{self.payload}\n[/UNTRUSTED CONTENT]"
            return injected_obs
            
        return original_observation
