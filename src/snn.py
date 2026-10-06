import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuração
# ============================================================

np.random.seed(42)

N_INPUT = 4
N_OUTPUT = 3

T = 60                  # passos por apresentação
N_EPOCHS = 200

BETA = 0.90
THRESHOLD = 1.0

LR = 0.03

TAU_PLUS = 10.0
TAU_MINUS = 10.0

A_PLUS = 1.0
A_MINUS = 0.8

W_MIN = 0.0
W_MAX = 1.0


# ============================================================
# Neurônio LIF
# ============================================================

class LIFNeuron:

    def __init__(self, beta=0.9, threshold=1.0):
        self.beta = beta
        self.threshold = threshold

        self.v = 0.0
        self.spike = 0

    def reset(self):
        self.v = 0.0
        self.spike = 0

    def step(self, current):

        # integração + leak
        self.v = self.beta * self.v + current

        # threshold
        if self.v >= self.threshold:
            self.spike = 1
            self.v = 0.0
        else:
            self.spike = 0

        return self.spike


# ============================================================
# Camada
# ============================================================

class SNN:

    def __init__(self):

        self.neurons = [
            LIFNeuron(BETA, THRESHOLD)
            for _ in range(N_OUTPUT)
        ]

        # pesos:
        #
        # input -> output
        #
        # shape:
        # [N_OUTPUT, N_INPUT]

        self.weights = np.random.uniform(
            0.2,
            0.8,
            size=(N_OUTPUT, N_INPUT)
        )

        # último spike de cada input
        self.last_pre_spike = np.full(
            N_INPUT,
            -np.inf
        )

        # último spike de cada output
        self.last_post_spike = np.full(
            N_OUTPUT,
            -np.inf
        )

    def reset_state(self):

        for neuron in self.neurons:
            neuron.reset()

        self.last_pre_spike[:] = -np.inf
        self.last_post_spike[:] = -np.inf


    # ========================================================
    # STDP
    # ========================================================

    def stdp_post(self, neuron_id, t):

        """
        Chamado quando o neurônio pós-sináptico dispara.

        PRE antes de POST:
            LTP
        """

        for i in range(N_INPUT):

            if np.isfinite(self.last_pre_spike[i]):

                dt = t - self.last_pre_spike[i]

                if dt > 0:

                    dw = (
                        LR
                        * A_PLUS
                        * np.exp(-dt / TAU_PLUS)
                    )

                    self.weights[neuron_id, i] += dw


    def stdp_pre(self, input_id, t):

        """
        Chamado quando um input dispara.

        POST antes de PRE:
            LTD
        """

        for j in range(N_OUTPUT):

            if np.isfinite(self.last_post_spike[j]):

                dt = t - self.last_post_spike[j]

                if dt > 0:

                    dw = (
                        LR
                        * A_MINUS
                        * np.exp(-dt / TAU_MINUS)
                    )

                    self.weights[j, input_id] -= dw


    # ========================================================
    # Simulação
    # ========================================================

    def run(self, input_spikes, learn=True):

        self.reset_state()

        output_spikes = np.zeros(
            (T, N_OUTPUT)
        )

        voltages = np.zeros(
            (T, N_OUTPUT)
        )

        for t in range(T):

            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------

            pre = input_spikes[t]

            # STDP causado por spikes PRE
            if learn:

                for i in range(N_INPUT):

                    if pre[i]:

                        self.stdp_pre(i, t)

                        self.last_pre_spike[i] = t

            else:

                for i in range(N_INPUT):

                    if pre[i]:
                        self.last_pre_spike[i] = t


            # ------------------------------------------------
            # Corrente sináptica
            # ------------------------------------------------

            currents = self.weights @ pre


            # ------------------------------------------------
            # Atualização dos neurônios
            # ------------------------------------------------

            candidate_spikes = []

            for j, neuron in enumerate(self.neurons):

                spike = neuron.step(
                    currents[j]
                )

                voltages[t, j] = neuron.v

                if spike:
                    candidate_spikes.append(j)


            # ------------------------------------------------
            # Winner Take All
            # ------------------------------------------------

            winner = None

            if candidate_spikes:

                # escolhemos quem recebeu maior corrente

                winner = max(
                    candidate_spikes,
                    key=lambda j: currents[j]
                )

                # cancelamos spikes dos outros
                for j in candidate_spikes:

                    if j != winner:
                        self.neurons[j].spike = 0


                output_spikes[t, winner] = 1


                # --------------------------------------------
                # STDP
                # --------------------------------------------

                if learn:

                    self.stdp_post(
                        winner,
                        t
                    )

                self.last_post_spike[winner] = t


            # manter pesos válidos

            self.weights = np.clip(
                self.weights,
                W_MIN,
                W_MAX
            )

        return output_spikes, voltages


# ============================================================
# Geração dos padrões
# ============================================================

def generate_pattern(pattern_id):

    """
    Três padrões diferentes.

    Cada padrão ativa preferencialmente
    um conjunto diferente de inputs.
    """

    spikes = np.zeros(
        (T, N_INPUT),
        dtype=int
    )

    if pattern_id == 0:

        active = [0, 1]

    elif pattern_id == 1:

        active = [1, 2]

    else:

        active = [2, 3]


    # spikes aproximadamente periódicos

    for i in active:

        for t in range(
            5,
            T,
            8
        ):

            jitter = np.random.randint(
                -1,
                2
            )

            ts = t + jitter

            if 0 <= ts < T:
                spikes[ts, i] = 1


    # ruído

    noise = (
        np.random.random(
            (T, N_INPUT)
        ) < 0.01
    )

    spikes = np.logical_or(
        spikes,
        noise
    ).astype(int)

    return spikes


# ============================================================
# Criar rede
# ============================================================

network = SNN()


print("\nPesos iniciais:\n")
print(network.weights)


# ============================================================
# Treinamento
# ============================================================

weight_history = []

for epoch in range(N_EPOCHS):

    pattern_id = np.random.randint(
        0,
        3
    )

    input_spikes = generate_pattern(
        pattern_id
    )

    network.run(
        input_spikes,
        learn=True
    )

    weight_history.append(
        network.weights.copy()
    )


weight_history = np.array(
    weight_history
)


print("\nPesos finais:\n")
print(network.weights)


# ============================================================
# Teste
# ============================================================

test_pattern = 0

input_spikes = generate_pattern(
    test_pattern
)

output_spikes, voltages = network.run(
    input_spikes,
    learn=False
)


print(
    "\nTeste do padrão:",
    test_pattern
)

print(
    "Spikes de saída:",
    output_spikes.sum(axis=0)
)


# ============================================================
# Raster plot INPUT
# ============================================================

plt.figure(figsize=(12, 4))

for t in range(T):

    for neuron in range(N_INPUT):

        if input_spikes[t, neuron]:

            plt.scatter(
                t,
                neuron,
                marker="|",
                s=150
            )

plt.xlabel("Tempo")
plt.ylabel("Input neuron")
plt.title("Input spike raster")

plt.show()


# ============================================================
# Raster OUTPUT
# ============================================================

plt.figure(figsize=(12, 4))

for t in range(T):

    for neuron in range(N_OUTPUT):

        if output_spikes[t, neuron]:

            plt.scatter(
                t,
                neuron,
                marker="|",
                s=150
            )

plt.xlabel("Tempo")
plt.ylabel("Output neuron")
plt.title("Output spike raster")

plt.show()


# ============================================================
# Evolução dos pesos
# ============================================================

plt.figure(figsize=(12, 6))

for neuron in range(N_OUTPUT):

    for input_id in range(N_INPUT):

        plt.plot(
            weight_history[:, neuron, input_id],
            label=f"N{neuron} <- I{input_id}"
        )

plt.xlabel("Epoch")
plt.ylabel("Weight")

plt.title(
    "Evolução dos pesos sinápticos"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.show()


# ============================================================
# Potencial de membrana
# ============================================================

plt.figure(figsize=(12, 5))

for neuron in range(N_OUTPUT):

    plt.plot(
        voltages[:, neuron],
        label=f"N{neuron}"
    )

plt.xlabel("Tempo")
plt.ylabel("Membrane potential")

plt.title(
    "Potencial de membrana"
)

plt.legend()

plt.show()