import torch
from record_utils import untuple_tensor


class HookWithCountThreshold:
    def __init__(self, steer_vec, scale, threshold=3, normalize=True):
        """
        Args:
            tokenizer: A tokenizer with a .decode() method to convert token IDs to strings.
            threshold (int): The occurrence count at which to trigger the hook.
        """
        self.steer_vec = steer_vec
        self.scale = scale
        self.threshold = threshold
        if normalize:
            self.steer_vec = self.steer_vec / self.steer_vec.norm()
        self.counter = 0

    def __call__(self, module, input, output):
        """
        The forward hook that checks the output of a module.
        It assumes that the output is either:
          - a 3D tensor: (batch_size, seq_length, vocab_size) or
          - a 2D tensor: (batch_size, vocab_size)
        """
        self.counter += 1
        if self.counter < self.threshold:
            return output

        print("Intervening!!")
        acts = untuple_tensor(output).clone().detach()
        if acts.dim() != 3:
            print("Hmm..")
            breakpoint()

        acts = acts[:, -1, :] + (self.scale * self.steer_vec)
        try:
            output[0][:, -1, :] = acts
        except:
            breakpoint()
        return output
