# Probe_Based_ML_Codex

## Project story (post-meeting, pre-implementation)

### 1. Motivation and research context
Reconfigurable Intelligent Surfaces (RIS) are a key enabling technology for future wireless systems because they can shape the wireless propagation environment by applying programmable phase shifts to many passive elements.

In an RIS-assisted link, the received signal power at the user depends on:
- the channel between the transmitter and the RIS,
- the channel between the RIS and the receiver,
- and the phase configuration applied at the RIS.

From a theoretical perspective, if the full channel state information (CSI) is known, the RIS phase vector can be designed to coherently combine all reflected paths and maximize the received power.

However, in realistic deployments:
- RISs are passive or semi-passive,
- acquiring full CSI is difficult or impractical,
- and RIS controllers must operate with limited information and low complexity.

This creates a gap between theoretical optimal RIS control and what is realistically achievable.

### 2. Role of verified analytical models
Before introducing any learning component, the work is grounded in verified physical models, not just simulations or ML heuristics. The RIS link-level behavior is described using established analytical received-power models, which differ depending on assumptions such as:
- far-field vs near-field propagation,
- element-wise modeling vs beamforming approximations,
- geometric configuration and distance scaling.

During the meeting, several models were explicitly referenced and compared conceptually:
- **General RIS channel model (Model A):** a physically general expression where the received signal is written as a coherent sum over RIS elements, explicitly accounting for the transmitter–RIS channel and RIS–receiver channel on a per-element basis.
- **Far-field beamforming-based model (Model B):** a simplified model assuming far-field conditions, where the RIS behaves like a reflecting array forming a beam. This model is accurate only when distances are sufficiently large and angular spreads are small.
- **Alternative general formulation (Model C):** another general model derived under slightly different assumptions, often appearing in recent RIS literature, and sometimes leading to different distance scaling laws.
- **Experimental measurements (reference):** published measurement-based curves that show how received power behaves in real setups and reveal which analytical model best matches reality under specific conditions.

The supervisor emphasized that:
- these models do not always coincide,
- and part of the scientific rigor of the PhD is to anchor all learning-based methods to one clearly defined, physically justified model.

Therefore, machine learning is not introduced until the analytical baseline is fully understood and fixed.

### 3. Core problem definition (agreed)
After clarifying the modeling foundation, the supervisor and student agreed on the following core control problem:

**The RIS must be configured to maximize received power without assuming access to full channel state information, and under constraints compatible with practical RIS hardware.**

This immediately rules out:
- direct use of analytical optimal phase formulas in operation,
- continuous phase optimization relying on full CSI.

Instead, the RIS controller must rely on indirect information.

### 4. Why probe-based control was chosen as the first step
Rather than jumping directly to predicting a full RIS phase vector, the supervisor proposed starting with a probe-based formulation, which is both:
- physically interpretable,
- and practically feasible.

The agreed idea is:
- define a finite set of RIS phase configurations, called probes,
- each probe is a valid RIS phase vector,
- for a fixed channel realization, different probes lead to different received powers,
- among the probe set, one probe yields the maximum received power.

This reframes RIS control as a selection problem, not a continuous optimization problem.

Key reasons this was chosen:
- probing a limited number of configurations is realistic in practice,
- it avoids explicit channel estimation,
- it allows clear comparison with an oracle reference.

### 5. Definition of “optimal” and performance metric
For each channel realization:
- the best probe is defined as the one that yields the highest received power among the probe set,
- this best probe is not the global analytical optimum, but a constrained optimum within the probe bank.

The agreed performance metric is the power ratio:

\[
\eta = \frac{P_{\text{selected}}}{P_{\text{best probe}}}
\]

where:
- \(P_{\text{best probe}}\) is the received power of the best probe (oracle),
- \(P_{\text{selected}}\) is the received power of the probe selected by the learning system.

This metric:
- is dimensionless,
- is bounded by 1,
- and provides a clean, interpretable measure of suboptimality.

### 6. Learning problem formulation (conceptual)
With this setup, the agreed machine learning task is:
- **Input:** indirect observations derived from probing or related measurements.
- **Output:** a decision indicating which probe to select.
- **Objective:** maximize \(\eta\), i.e., select the probe that gives power as close as possible to the best probe.

Importantly:
- the ML model is not learning the channel,
- it is not learning the analytical formula,
- it is learning a decision rule that exploits structure implicitly present in the physics.

### 7. Scientific purpose of this stage
This stage is not the final goal, but a validation step. It is meant to answer a fundamental question:

**Can a learning-based controller, without channel knowledge, reliably select RIS configurations that are near-optimal relative to a physically defined baseline?**

If the answer is yes, then it justifies:
- moving to richer control strategies,
- considering adaptive or sequential probing,
- and eventually predicting or refining RIS phase configurations directly.

If the answer is no, then more sensing or structure is required.

### 8. Agreed scope limit (important)
At this stage, the project does not consider:
- multi-user systems,
- network-level optimization,
- temporal dynamics or mobility,
- hardware non-idealities.

The scope is deliberately narrow to ensure:
- physical correctness,
- interpretability,
- and reproducibility.

### 9. Probe definition (confirmed)
The probes are defined as a **structured, orthogonal set of phase configurations**. This means the probe bank is systematic rather than random, enabling clear interpretation and consistent comparison across channel realizations.
