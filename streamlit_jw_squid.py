# --------------------------------------------
#  Streamlit: Squid‑side‑game incentive tool
#  Default profile “Wesley” (N+2, squid = 2.5)
#  Supports N+2, N+3, N+4, Dino, Infinite
# --------------------------------------------
import streamlit as st
import functools
import math
import pandas as pd

st.set_page_config(layout="wide")

###############################################################################
# SESSION STATE
###############################################################################
if "results_table" not in st.session_state:
    st.session_state["results_table"] = None

###############################################################################
# 1) TOP‑ROW INPUTS — (Game Mode) (Squid Value) (Threshold Controls)
###############################################################################
top_col1, top_col2, top_col3 = st.columns([1, 1, 1])

# ――― Side‑game format — default = N+2 (“Wesley”) ―――
with top_col1:
    game_mode = st.selectbox(
        "Side‑Game Format",
        ["N+2", "N+3", "N+4", "Dino", "Infinite"],
        index=0,
        key="game_mode_select"
    )

# ――― Squid value — default = 2.5 ―――
with top_col2:
    squid_value = st.number_input(
        "Squid Value",
        value=2.5,
        step=0.5,
        min_value=0.0,
        key="squid_value_input"
    )

# ――― Threshold profile dropdown (N+X modes only) ―――
threshold_map = {}
if game_mode in ["N+2", "N+3", "N+4"]:
    with top_col3:
        defaults_options = [
            "Wesley",                          # new default profile
            "N+3, 1/1/(3) 2x@3",
            "1/2 5 2x@3",
            "1/1 3 2x@3 3x@5",
            "Tui",
            "None",
        ]
        chosen_default = st.selectbox("Defaults", defaults_options, index=0)

        # Suggested starting number of thresholds
        if chosen_default == "Wesley":
            initial_thr = 1
        elif chosen_default == "N+3, 1/1/(3) 2x@3":
            initial_thr = 1
        elif chosen_default == "1/2 5 2x@3":
            initial_thr = 1
        elif chosen_default == "1/1 3 2x@3 3x@5":
            initial_thr = 2
        elif chosen_default == "Tui":
            initial_thr = 4
        else:
            initial_thr = 1

        num_thr = st.number_input(
            "How many thresholds?",
            value=initial_thr,
            min_value=0,
            step=1,
            key="num_thr_box"
        )

        for i in range(int(num_thr)):
            # Pick sensible defaults based on the chosen profile
            if chosen_default == "Wesley":
                default_h, default_m = (3, 2.0)
            elif chosen_default == "N+3, 1/1/(3) 2x@3":
                default_h, default_m = ((3, 2.0) if i == 0 else (1, 1.0))
            elif chosen_default == "1/2 5 2x@3":
                default_h, default_m = ((3, 2.0) if i == 0 else (1, 1.0))
            elif chosen_default == "1/1 3 2x@3 3x@5":
                default_h, default_m = (
                    (3, 2.0) if i == 0 else
                    (5, 3.0) if i == 1 else
                    (1, 1.0)
                )
            elif chosen_default == "Tui":
                default_h, default_m = (
                    (3, 2.0) if i == 0 else
                    (5, 3.0) if i == 1 else
                    (7, 4.0) if i == 2 else
                    (9, 5.0) if i == 3 else
                    (1, 1.0)
                )
            else:  # "None"
                default_h, default_m = (3, 2.0)

            cx, cy = st.columns([1, 1])
            h_val = cx.number_input(
                f"h for threshold {i + 1}",
                value=default_h,
                step=1,
                min_value=1,
                key=f"th_h_{i}"
            )
            m_val = cy.number_input(
                f"Multiplier at h = {h_val}",
                value=default_m,
                step=1.0,
                key=f"th_m_{i}"
            )
            threshold_map[h_val] = m_val
else:
    threshold_map = {}

###############################################################################
# 2) NUMBER OF PLAYERS
###############################################################################
N = st.number_input(
    "Number of Players (N)",
    value=8,
    min_value=1,
    step=1
)

###############################################################################
# 3) HORIZONTAL LINE
###############################################################################
st.markdown(
    "<hr style='margin:0; height:1px; background-color:#ccc; border:none;' />",
    unsafe_allow_html=True
)

###############################################################################
# 4) CURRENT TOKEN DISTRIBUTION ENTRY GRID
###############################################################################
def get_seat_labels(num_seats: int):
    """Return seat labels ending BTN, SB, BB, S for last four."""
    if num_seats < 4:
        return [f"Player{i + 1}" for i in range(num_seats)]
    neg_count = num_seats - 4
    neg_lbls = [f"-{x}" for x in range(neg_count, 0, -1)]
    return (neg_lbls + ["BTN", "SB", "BB", "S"])[:num_seats]

seat_labels = get_seat_labels(N)
h_current = []

if game_mode == "Dino":
    for i in range(N):
        v = st.radio(f"{seat_labels[i]}", [0, 1], horizontal=True,
                     index=0, key=f"dino_{i}")
        h_current.append(v)
else:
    for i in range(N):
        v = st.number_input(
            f"{seat_labels[i]} tokens",
            value=0,
            step=1,
            min_value=0,
            key=f"dist_{i}"
        )
        h_current.append(v)

###############################################################################
# 5) BACK‑END MATH
###############################################################################
def seat_value_nplusX(h, thr_map_tuple, sq_val):
    thr_map = dict(thr_map_tuple)
    mm = 1.0
    for (th, mul) in sorted(thr_map.items()):
        if h >= th and mul > mm:
            mm = mul
    return h * mm * sq_val

def payoff_singled_out_nplusX(h_list, zero_idx, thr_map_tuple, sq_val):
    n_len = len(h_list)
    seatvals = [seat_value_nplusX(h, thr_map_tuple, sq_val) for h in h_list]
    total_val = sum(seatvals)
    out = [0] * n_len
    out[zero_idx] = -total_val
    for i in range(n_len):
        if i != zero_idx:
            out[i] = seatvals[i]
    return tuple(out)

def payoff_multiple_zero_nplusX(h_list, thr_map_tuple, sq_val):
    n_len = len(h_list)
    seatvals = [seat_value_nplusX(h, thr_map_tuple, sq_val) for h in h_list]
    total_val = sum(seatvals)
    zc = sum(x == 0 for x in h_list)
    out = [0] * n_len
    for i in range(n_len):
        if h_list[i] == 0:
            out[i] = -total_val
        else:
            out[i] = seatvals[i] * zc
    return tuple(out)

def make_compute_ev_nplusX(extra_tokens: int):
    @functools.lru_cache(None)
    def _compute_ev(h_tuple, T_left, p_tuple, n_len,
                    thr_map_tuple, sq_val, first_pot):
        h_list = list(h_tuple)
        zc = sum(x == 0 for x in h_list)

        # Terminal checks
        if T_left == 0:
            if zc == 0:
                return (0,) * n_len
            if zc == 1:
                iz = [i for i, x in enumerate(h_list) if x == 0][0]
                return payoff_singled_out_nplusX(h_list, iz,
                                                 thr_map_tuple, sq_val)
            return payoff_multiple_zero_nplusX(h_list, thr_map_tuple, sq_val)

        if zc == 1:
            iz = [i for i, x in enumerate(h_list) if x == 0][0]
            return payoff_singled_out_nplusX(h_list, iz,
                                             thr_map_tuple, sq_val)

        out = [0] * n_len
        distribution = p_tuple if first_pot else [1.0 / n_len] * n_len

        for w in range(n_len):
            pw = distribution[w]
            if pw > 0:
                new_h = h_list[:]
                new_h[w] += 1
                ev_sub = _compute_ev(
                    tuple(new_h),
                    T_left - 1,
                    p_tuple,
                    n_len,
                    thr_map_tuple,
                    sq_val,
                    False
                )
                for i in range(n_len):
                    out[i] += pw * ev_sub[i]
        return tuple(out)

    return _compute_ev

compute_ev_Nplus2 = make_compute_ev_nplusX(2)
compute_ev_Nplus3 = make_compute_ev_nplusX(3)
compute_ev_Nplus4 = make_compute_ev_nplusX(4)

# --- Dino mode helpers ------------------------------------------------------
def payoff_singled_out_dino(h_list, zero_idx, sq_val):
    n_len = len(h_list)
    holders = sum(x == 1 for x in h_list)
    total_val = holders * sq_val
    out = [0] * n_len
    out[zero_idx] = -total_val
    for i in range(n_len):
        if i != zero_idx and h_list[i] == 1:
            out[i] = sq_val
    return tuple(out)

@functools.lru_cache(None)
def compute_ev_Dino(h_tuple, p_tuple, n_len, sq_val, first_pot):
    h_list = list(h_tuple)
    zc = sum(x == 0 for x in h_list)
    if zc == 1:
        iz = [i for i, x in enumerate(h_list) if x == 0][0]
        return payoff_singled_out_dino(h_list, iz, sq_val)
    if zc == 0:
        return (0,) * n_len

    distribution = p_tuple if first_pot else [1.0 / n_len] * n_len
    probH = sum(distribution[i] for i, h in enumerate(h_list) if h == 1)
    probZ = 1.0 - probH
    if probH >= 1.0 or probZ <= 0:
        return (0,) * n_len

    sumEV = [0] * n_len
    zero_idx = [i for i, h in enumerate(h_list) if h == 0]
    for zpos in zero_idx:
        relw = distribution[zpos] / probZ
        new_h = h_list[:]
        new_h[zpos] = 1
        ev_sub = compute_ev_Dino(tuple(new_h), p_tuple, n_len, sq_val, False)
        for k in range(n_len):
            sumEV[k] += relw * ev_sub[k]
    factor = probZ / (1.0 - probH)
    return tuple(factor * x for x in sumEV)

# --- Infinite mode helpers --------------------------------------------------
def payoff_singled_out_infinite(h_list, zero_idx, sq_val):
    n_len = len(h_list)
    total_t = sum(h_list)
    total_val = total_t * sq_val
    out = [0] * n_len
    out[zero_idx] = -total_val
    for i in range(n_len):
        if i != zero_idx:
            out[i] = h_list[i] * sq_val
    return tuple(out)

@functools.lru_cache(None)
def compute_ev_Infinite(h_tuple, p_tuple, n_len, sq_val, first_pot):
    h_list = list(h_tuple)
    zc = sum(x == 0 for x in h_list)
    if zc == 1:
        iz = [i for i, x in enumerate(h_list) if x == 0][0]
        return payoff_singled_out_infinite(h_list, iz, sq_val)
    if zc == 0:
        return (0,) * n_len

    distribution = p_tuple if first_pot else [1.0 / n_len] * n_len
    probH = sum(distribution[i] for i, h in enumerate(h_list) if h > 0)
    probZ = 1.0 - probH
    if probH >= 1.0 or probZ <= 0:
        return (0,) * n_len

    sumEV = [0] * n_len
    zero_idx = [i for i, h in enumerate(h_list) if h == 0]
    for zpos in zero_idx:
        relw = distribution[zpos] / probZ
        new_h = h_list[:]
        new_h[zpos] += 1
        ev_sub = compute_ev_Infinite(tuple(new_h), p_tuple,
                                     n_len, sq_val, False)
        for k in range(n_len):
            sumEV[k] += relw * ev_sub[k]
    factor = probZ / (1.0 - probH)
    return tuple(factor * x for x in sumEV)

###############################################################################
# 6) SCENARIO‑LEVEL WRAPPERS
###############################################################################
def scenario_ev_R(h_tuple, mode, p_tuple, thr_map_tuple,
                  sq_val, first_pot=True):
    n_len = len(h_tuple)
    if mode == "N+2":
        T_left = (n_len + 2) - sum(h_tuple)
        return compute_ev_Nplus2(h_tuple, T_left, p_tuple,
                                 n_len, thr_map_tuple, sq_val, first_pot)
    if mode == "N+3":
        T_left = (n_len + 3) - sum(h_tuple)
        return compute_ev_Nplus3(h_tuple, T_left, p_tuple,
                                 n_len, thr_map_tuple, sq_val, first_pot)
    if mode == "N+4":
        T_left = (n_len + 4) - sum(h_tuple)
        return compute_ev_Nplus4(h_tuple, T_left, p_tuple,
                                 n_len, thr_map_tuple, sq_val, first_pot)
    if mode == "Dino":
        return compute_ev_Dino(h_tuple, p_tuple, n_len, sq_val, first_pot)
    # Infinite
    return compute_ev_Infinite(h_tuple, p_tuple, n_len, sq_val, first_pot)

def scenario_ev_A(h_tuple, mode, p_tuple, seat_i,
                  thr_map_tuple, sq_val, first_pot=True):
    h_list = list(h_tuple)
    n_len = len(h_list)

    # (1) Token goes to HERO
    if mode == "N+2":
        T_left = (n_len + 2) - sum(h_list)
        if T_left <= 0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple,
                                 thr_map_tuple, sq_val, False)
        h_list[seat_i] += 1
        return compute_ev_Nplus2(tuple(h_list), T_left - 1, p_tuple,
                                 n_len, thr_map_tuple, sq_val, False)

    if mode == "N+3":
        T_left = (n_len + 3) - sum(h_list)
        if T_left <= 0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple,
                                 thr_map_tuple, sq_val, False)
        h_list[seat_i] += 1
        return compute_ev_Nplus3(tuple(h_list), T_left - 1, p_tuple,
                                 n_len, thr_map_tuple, sq_val, False)

    if mode == "N+4":
        T_left = (n_len + 4) - sum(h_list)
        if T_left <= 0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple,
                                 thr_map_tuple, sq_val, False)
        h_list[seat_i] += 1
        return compute_ev_Nplus4(tuple(h_list), T_left - 1, p_tuple,
                                 n_len, thr_map_tuple, sq_val, False)

    if mode == "Dino":
        if h_list[seat_i] == 0:
            h_list[seat_i] = 1
        return compute_ev_Dino(tuple(h_list), p_tuple, n_len, sq_val, False)

    # Infinite
    h_list[seat_i] += 1
    return compute_ev_Infinite(tuple(h_list), p_tuple,
                               n_len, sq_val, False)

def scenario_ev_B(h_tuple, mode, p_tuple, seat_i,
                  thr_map_tuple, sq_val, first_pot=True):
    h_list = list(h_tuple)
    n_len = len(h_list)
    sum_others = 1.0 - p_tuple[seat_i]
    if sum_others <= 0:
        return scenario_ev_R(tuple(h_list), mode, p_tuple,
                             thr_map_tuple, sq_val, False)

    def aggregate_for_mode(compute_ev_fn, T_left):
        accum = [0] * n_len
        for other in range(n_len):
            if other == seat_i:
                continue
            wprob = p_tuple[other] / sum_others
            alt_h = h_list[:]
            alt_h[other] += 1
            ev_sub = compute_ev_fn(
                tuple(alt_h), T_left - 1, p_tuple,
                n_len, thr_map_tuple, sq_val, False
            )
            for k in range(n_len):
                accum[k] += wprob * ev_sub[k]
        return tuple(accum)

    if mode == "N+2":
        T_left = (n_len + 2) - sum(h_list)
        if T_left <= 0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple,
                                 thr_map_tuple, sq_val, False)
        return aggregate_for_mode(compute_ev_Nplus2, T_left)

    if mode == "N+3":
        T_left = (n_len + 3) - sum(h_list)
        if T_left <= 0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple,
                                 thr_map_tuple, sq_val, False)
        return aggregate_for_mode(compute_ev_Nplus3, T_left)

    if mode == "N+4":
        T_left = (n_len + 4) - sum(h_list)
        if T_left <= 0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple,
                                 thr_map_tuple, sq_val, False)
        return aggregate_for_mode(compute_ev_Nplus4, T_left)

    if mode == "Dino":
        accum = [0] * n_len
        for other in range(n_len):
            if other == seat_i:
                continue
            wprob = p_tuple[other] / sum_others
            alt_h = h_list[:]
            if alt_h[other] == 0:
                alt_h[other] = 1
            ev_sub = compute_ev_Dino(tuple(alt_h), p_tuple,
                                     n_len, sq_val, False)
            for k in range(n_len):
                accum[k] += wprob * ev_sub[k]
        return tuple(accum)

    # Infinite
    accum = [0] * n_len
    for other in range(n_len):
        if other == seat_i:
            continue
        wprob = p_tuple[other] / sum_others
        alt_h = h_list[:]
        alt_h[other] += 1
        ev_sub = compute_ev_Infinite(tuple(alt_h), p_tuple,
                                     n_len, sq_val, False)
        for k in range(n_len):
            accum[k] += wprob * ev_sub[k]
    return tuple(accum)

###############################################################################
# 7) FRONT‑END — “Compute Incentives” BUTTON
###############################################################################
def do_compute_incentives():
    p_tuple = tuple([1.0 / N] * N)
    h_tuple = tuple(h_current)
    thr_tup = tuple(sorted(threshold_map.items()))

    results = []
    for seat_i in range(N):
        evA = scenario_ev_A(h_tuple, game_mode, p_tuple,
                            seat_i, thr_tup, squid_value, True)
        evB = scenario_ev_B(h_tuple, game_mode, p_tuple,
                            seat_i, thr_tup, squid_value, True)
        myA, myB = evA[seat_i], evB[seat_i]
        results.append({
            "Seat": seat_labels[seat_i],
            "EV(A)": round(myA, 2),
            "EV(B)": round(myB, 2),
            "Incentive (A‑B)": round(myA - myB, 2),
            "Tokens": h_current[seat_i],
        })
    return results

compute_btn = st.button("Compute Incentives Now")
if compute_btn:
    new_table = do_compute_incentives()
    st.session_state["results_table"] = new_table

if st.session_state["results_table"] is not None:
    combined_df = pd.DataFrame(st.session_state["results_table"])
    combined_df.set_index("Seat", inplace=True)
    st.dataframe(combined_df, use_container_width=True)
else:
    st.info("Results will appear here after “Compute Incentives Now”.")
