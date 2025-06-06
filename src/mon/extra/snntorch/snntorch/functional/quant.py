import torch
import math


class StateQuant(torch.autograd.Function):
    """Wrapper function for state_quant"""

    @staticmethod
    def forward(ctx, input_, levels):

        device = input_.device
        levels = levels.to(device)
        size = input_.size()
        input_ = input_.flatten()

        # Broadcast mem along new direction same # of times as num_levels
        repeat_dims = torch.ones(len(input_.size())).tolist()
        repeat_dims.append(len(levels))
        repeat_dims = [int(item) for item in repeat_dims]
        repeat_dims = tuple(repeat_dims)
        input_ = input_.unsqueeze(-1).repeat(repeat_dims)

        # find closest valid quant state
        idx_match = torch.min(torch.abs(levels - input_), dim=-1)[1]
        quant_tensor = levels[idx_match]

        return quant_tensor.reshape(size)

    # STE
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def state_quant(
    num_bits=8,
    uniform=True,
    thr_centered=True,
    threshold=1,
    lower_limit=0,
    upper_limit=0.2,
    multiplier=None,
):
    """Quantization-Aware Training with spiking neuron states.

    **Note: for weight quantization, we recommend using Brevitas or
    another pre-existing PyTorch-friendly library.**

    Uniform and non-uniform quantization can be applied in various
    modes by specifying ``uniform=True``.

    Valid quantization levels can be centered about 0 or threshold
    by specifying ``thr_centered=True``.

    ``upper_limit`` and ``lower_limit`` specify the proportion of how
    far valid levels go above and below the positive and negative threshold/
    E.g., upper_limit=0.2 means the maximum valid state is 20% higher
    than the value specified in ``threshold``.

    Example::

        import torch
        import snntorch as snn
        from snntorch.functional import quant

        beta = 0.5
        thr = 5

        # set the quantization parameters
        q_lif = quant.state_quant(num_bits=4, uniform=True, threshold=thr)

        # specifying state_quant applies state-quantization to the
        # hidden state(s) automatically
        lif = snn.Leaky(beta=beta, threshold=thr, state_quant=q_lif)

        rand_input = torch.rand(1)
        mem = lif.init_leaky()

        # forward-pass for one step
        spk, mem = lif(rand_input, mem)

    Note: Quantization-Aware training is focused on modelling a
    reduced precision network, but does not in of itself accelerate
    low-precision models.
    Hidden states are still represented as full precision values for
    compatibility with PyTorch.
    For accelerated performance or constrained-memory, the model should
    be exported to a downstream backend.


    :param num_bits: Number of bits to quantize state variables to,
        defaults to ``8``
    :type num_bits: int, optional

    :param uniform: Applies uniform quantization if specified, non-uniform
        if unspecified, defaults to ``True``
    :type uniform: Bool, optional

    :param thr_centered: For non-uniform quantization, specifies if valid
        states should be centered (densely clustered) around the threshold
        rather than at 0, defaults to ``True``
    :type thr_centered: Bool, optional

    :param threshold: Specifies the threshold, defaults to ``1``
    :type threshold: float, optional

    :param lower_limit: Specifies how far below (-threshold) the lowest
        valid state can be, i.e., (-threshold - threshold*lower_limit),
        defaults to ``0``
    :type lower_limit: float, optional

    :param upper_limit: Specifies how far above (threshold) the highest
        valid state can be, i.e., (threshold + threshold*upper_limit),
        defaults to ``0.2``
    :type upper_limit: float, optional

    :param multiplier: For non-uniform distributions, specify the base
        of the exponential. If ``None``, an appropriate value is set
        internally based on ``num_bits``, defaults to ``None``
    :type multiplier: float, optional

    """

    num_levels = 2 << num_bits - 1

    # linear / uniform quantization - ignores thr_centered
    if uniform:
        levels = torch.linspace(
            -threshold - threshold * lower_limit,
            threshold + threshold * upper_limit,
            num_levels,
        )

    # exponential / non-uniform quantization
    else:
        if multiplier is None:
            if num_bits == 1:
                multiplier = 0.05
            if num_bits == 2:
                multiplier = 0.1
            elif num_bits == 3:
                multiplier = 0.3
            elif num_bits == 4:
                multiplier = 0.5
            elif num_bits == 5:
                multiplier = 0.7
            elif num_bits == 6:
                multiplier = 0.9
            elif num_bits == 7:
                multiplier = 0.925
            elif num_bits > 7:
                multiplier = 0.95

        # asymmetric: shifted to threshold
        if thr_centered:
            max_val = threshold + (
                threshold * upper_limit
            )  # maximum level that can be reached
            min_val = -(
                threshold + (threshold * lower_limit)
            )  # minimum level that can be reached

            num_levels = 2 << num_bits - 1  # total number of levels

            overall_range = max_val - min_val  # range of number of levels

            lower_range = threshold - min_val  # levels below the threshold
            upper_range = max_val - threshold  # levels above the threshold

            lower_percent = (
                lower_range / overall_range
            )  # percent +66 the threshold
            upper_percent = (
                upper_range / overall_range
            )  # percent above the threshold

            lower_bits = math.floor(
                num_levels * lower_percent
            )  # how many levels lower than the threshold
            upper_bits = (
                num_levels - lower_bits
            )  # how many bits above the threshold

            lower_summation = 0
            store_val = []
            if lower_bits != 0:

                for j in reversed(
                    range(lower_bits)
                ):  # figure out how much the summation travels
                    lower_curr = multiplier ** j
                    lower_summation += multiplier ** j

                lower_room = lower_summation / lower_range
                min_temp_sum = min_val
                # store_val.append(min_temp_sum)

                for j in range(lower_bits):
                    lower_curr = multiplier ** j
                    store_val.append(min_temp_sum)
                    min_temp_sum += lower_curr / lower_room
                    # store_val.append(min_temp_sum)

            if upper_bits != 0:

                upper_summation = 0
                for j in reversed(range(upper_bits)):
                    upper_curr = multiplier ** j
                    upper_summation += multiplier ** j

                upper_room = upper_summation / upper_range

                max_temp_sum = threshold
                diff_store_val = []
                # store_val.append(max_temp_sum)
                for j in reversed(range(upper_bits)):
                    upper_curr = multiplier ** j
                    # store_val.append(max_temp_sum)
                    max_temp_sum += upper_curr / upper_room
                    store_val.append(max_temp_sum)

                # store_val.append(max_temp_sum)
            levels = torch.tensor([x for x in store_val])

        # centered about zero
        else:
            max_val = threshold + (
                threshold * upper_limit
            )  # maximum level that can be reached
            min_val = -(
                threshold + (threshold * lower_limit)
            )  # minimum level that can be reached

            num_levels = 2 << num_bits - 1  # total number of levels

            overall_range = max_val - min_val  # range of number of levels

            lower_range = 0 - min_val  # levels below the threshold
            upper_range = max_val - 0  # levels above the threshold

            lower_percent = (
                lower_range / overall_range
            )  # percent +66 the threshold
            upper_percent = (
                upper_range / overall_range
            )  # percent above the threshold

            lower_bits = math.floor(
                num_levels * lower_percent
            )  # how many levels lower than the threshold
            upper_bits = (
                num_levels - lower_bits
            )  # how many bits above the threshold

            lower_summation = 0
            store_val = []

            if lower_bits != 0:

                for j in reversed(
                    range(lower_bits)
                ):  # figure out how much the summation travels
                    lower_curr = multiplier ** j
                    lower_summation += multiplier ** j

                lower_room = lower_summation / lower_range

                min_temp_sum = min_val
                # store_val.append(min_temp_sum)

                for j in range(lower_bits):
                    lower_curr = multiplier ** j
                    store_val.append(min_temp_sum)
                    min_temp_sum += lower_curr / lower_room

                    # store_val.append(min_temp_sum)

            if upper_bits != 0:

                upper_summation = 0
                for j in reversed(range(upper_bits)):
                    upper_curr = multiplier ** j
                    upper_summation += multiplier ** j

                upper_room = upper_summation / upper_range

                max_temp_sum = 0
                diff_store_val = []
                # store_val.append(max_temp_sum)
                for j in reversed(range(upper_bits)):
                    upper_curr = multiplier ** j
                    max_temp_sum += upper_curr / upper_room
                    store_val.append(max_temp_sum)

                # store_val.append(max_temp_sum)

            levels = torch.tensor([x for x in store_val])

    def inner(x):
        return StateQuant.apply(x, levels)

    return inner
