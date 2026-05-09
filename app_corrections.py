"""
FlexiHome Audit Fixes: Priority 1, 2, 3 Corrections
=====================================================

This file documents all corrections needed for the MVP VPP application.
Apply these changes incrementally to app.py.

PRIORITY 1: CRITICAL FIXES (1 sprint)
=====================================
"""

# ============================================================================
# FIX 1: Result Source Tracking (Lines 2020-2033)
# ============================================================================
# BEFORE: Silent fallback without visibility
# upper_view_result = st.session_state.get("upper_mpc_result", {})
# if result_history(upper_view_result).empty:
#     upper_view_result = active_result  # ← No indication to user

# AFTER: Explicit source tracking
def get_mpc_result_with_source_tracking():
    """Retrieve and track MPC result source to prevent silent data corruption."""
    active_result = st.session_state.get(
        "mpc_result",
        {
            "history": pd.DataFrame(),
            "tracking_4s": pd.DataFrame(),
            "summary": {},
            "compliance": {},
            "first_schedule": pd.DataFrame(),
            "household_contributions": pd.DataFrame(),
            "appliance_contributions": pd.DataFrame(),
            "upper_device_schedule": pd.DataFrame(),
            "gateway_commands": pd.DataFrame(),
            "usage_fatigue_summary": pd.DataFrame(),
        },
    )
    
    upper_view_result = st.session_state.get("upper_mpc_result", {})
    upper_source = "upper_mpc_result (fresh)"
    
    if result_history(upper_view_result).empty:
        upper_view_result = active_result
        upper_source = "active_result (fallback: upper MPC empty)"
        if result_history(active_result).empty:
            upper_source = "empty (no MPC run yet)"
    
    lower_view_result = st.session_state.get("live_lower_result", {})
    lower_source = "live_lower_result (fresh)"
    
    if result_history(lower_view_result).empty and result_summary(active_result).get("execute_lower_mpc"):
        lower_view_result = active_result
        lower_source = "active_result (fallback: lower MPC empty)"
    
    # Store source tracking in session for later display
    st.session_state["_result_sources"] = {
        "upper": upper_source,
        "lower": lower_source,
        "active": "mpc_result"
    }
    
    return {
        "upper_view_result": upper_view_result,
        "lower_view_result": lower_view_result,
        "active_result": active_result,
        "sources": st.session_state["_result_sources"]
    }


# Usage in tabs[4] around line 2005:
# result_data = get_mpc_result_with_source_tracking()
# upper_view_result = result_data["upper_view_result"]
# lower_view_result = result_data["lower_view_result"]
# active_result = result_data["active_result"]
# sources = result_data["sources"]
#
# # Then display result source clearly:
# if result_history(active_result).empty:
#     st.info("Configure the scenario and click Run Upper MPC Layer...")
# else:
#     # Add source tracking display (line ~2054)
#     source_cols = st.columns(2)
#     source_cols[0].caption(f"📊 Upper result source: {sources['upper']}")
#     source_cols[1].caption(f"📊 Lower result source: {sources['lower']}")


# ============================================================================
# FIX 2: Comprehensive Session Reset on Scenario Change (Lines 1123-1130)
# ============================================================================
# BEFORE: Incomplete cleanup
# if next_scenario != st.session_state["scenario_inputs"]:
#     st.session_state["scenario_inputs"] = next_scenario
#     st.session_state["mpc_result_stale"] = True
#     st.session_state.pop("default_mpc_models", None)
#     # ... but missing upper_mpc_result, custom_mpc_models, forecasts, etc.

# AFTER: Comprehensive reset
LIVE_SESSION_KEYS_TO_CLEAR_ON_SCENARIO_CHANGE = [
    # Data bundle and results
    "portfolio_bundle",
    "portfolio_signature",
    "upper_mpc_result",
    "mpc_result",
    "mpc_config",
    
    # Live market state
    "live_market_session",
    "market_simulator",
    "live_market_ticks",
    "lower_mpc_armed",
    "live_lower_result",
    "lower_mpc_solving",  # New: lock state
    
    # Forecasting models (stale on scenario change)
    "default_mpc_models",
    "default_mpc_signature",
    "custom_mpc_models",
    
    # User-trained forecasts (no longer valid)
    "last_forecast_spec",
    "last_forecaster_instance",
    "last_forecast_comparison",
    "last_forecaster_plugin",
    "last_forecast_market_status",
    "last_forecast_row_count",
    
    # Source tracking
    "_result_sources",
    "_market_data_fallback_flag",
]

# Apply in scenario form (after line 1098):
# if apply_scenario:
#     next_scenario = { ... }
#     if next_scenario != st.session_state["scenario_inputs"]:
#         st.session_state["scenario_inputs"] = next_scenario
#         st.session_state["mpc_result_stale"] = True
#         
#         # Comprehensive reset
#         for key in LIVE_SESSION_KEYS_TO_CLEAR_ON_SCENARIO_CHANGE:
#             st.session_state.pop(key, None)
#         
#         st.rerun()


# ============================================================================
# FIX 3: Market Fallback Flag with User Warning (Lines 1188-1204)
# ============================================================================
# BEFORE: Silent fallback
# try:
#     bundle = generate_portfolio_runtime(...)
# except RuntimeError as exc:
#     st.error(f"Real market data could not be loaded: {exc}")
#     bundle = generate_portfolio_runtime(..., market_data_mode="synthetic")

# AFTER: Persistent fallback tracking
def generate_portfolio_with_fallback_tracking(
    portfolio_args: Dict,
    real_market_ready: bool,
    entsoe_key_input: str,
    fingrid_key_input: str,
) -> tuple:
    """
    Generate portfolio and track if real market data fell back to synthetic.
    Returns: (bundle, fallback_flag, fallback_reason)
    """
    fallback_flag = False
    fallback_reason = ""
    
    if not real_market_ready:
        # Synthetic from the start
        bundle = generate_portfolio_runtime(
            **portfolio_args,
            market_data_mode="synthetic",
            use_cache=True,
        )
        return bundle, False, ""
    
    try:
        bundle = generate_portfolio_runtime(
            **portfolio_args,
            market_data_mode="real",
            entsoe_api_key=entsoe_key_input.strip() or None,
            fingrid_api_key=fingrid_key_input.strip() or None,
            use_cache=False,
        )
        return bundle, False, ""
    except RuntimeError as exc:
        st.error(f"🔴 Real market API failed: {exc}")
        fallback_reason = str(exc)
        fallback_flag = True
        
        with st.spinner("Falling back to synthetic market data..."):
            bundle = generate_portfolio_runtime(
                **portfolio_args,
                market_data_mode="synthetic",
                use_cache=True,
            )
        
        return bundle, True, fallback_reason


# Usage (around line 1188):
# bundle, market_fallback, fallback_reason = generate_portfolio_with_fallback_tracking(
#     portfolio_args,
#     real_market_ready,
#     entsoe_key_input,
#     fingrid_key_input,
# )
# st.session_state["_market_data_fallback_flag"] = market_fallback
# st.session_state["_market_fallback_reason"] = fallback_reason

# Then display warning throughout app (e.g., in tabs[1]):
# if st.session_state.get("_market_data_fallback_flag"):
#     st.warning(
#         f"⚠️ **Currently using SYNTHETIC market data** due to API failure: {st.session_state.get('_market_fallback_reason', 'Unknown')}\n\n"
#         f"Prices, activation signals, and forecasts are NOT from real markets. "
#         f"Compliance audit results may not reflect production accuracy."
#     )


# ============================================================================
# FIX 4: Fragment Cache Key Fix - Use elapsed_tick (Line 429 & 1939)
# ============================================================================
# BEFORE: Cache expires every ~4 seconds causing thrashing
# @st.cache_data(show_spinner=False, ttl=4)
# def cached_lower_mpc_from_upper(..., elapsed_tick: int, ...):
#     elapsed_seconds = float(max(int(elapsed_tick), 0) * max(int(inner_dt_seconds), 1))
#     return run_lower_mpc_from_upper_result(...)

# AFTER: Cache only misses when elapsed_tick actually changes
@st.cache_data(show_spinner=False)
def cached_lower_mpc_from_upper_fixed(
    df: pd.DataFrame,
    upper_result: Dict[str, object],
    fleet_meta: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    preview_4s: pd.DataFrame,
    device_roster: pd.DataFrame,
    elapsed_tick: int,  # ← Use as cache differentiator
    inner_dt_seconds: int = 4,
) -> Dict[str, object]:
    """
    Cached lower MPC solver. Cache key includes elapsed_tick,
    so it only re-solves when time actually advances.
    """
    elapsed_seconds = float(max(int(elapsed_tick), 0) * max(int(inner_dt_seconds), 1))
    return run_lower_mpc_from_upper_result(
        df=df,
        upper_result=upper_result,
        fleet_meta=fleet_meta,
        market_mode=market_mode,
        resource_mode=resource_mode,
        preview_4s=preview_4s,
        device_roster=device_roster,
        inner_controller_mode="mpc",
        inner_dt_seconds=int(inner_dt_seconds),
        inner_mpc_horizon_seconds=4,
        rotation_strategy="usage_aware",
        gateway_mode="live_market_coupled",
        elapsed_seconds=elapsed_seconds,
    )

# Replace line 429 with:
# @st.cache_data(show_spinner=False)  # ← Remove ttl=4
# def cached_lower_mpc_from_upper(...):


# ============================================================================
# PRIORITY 2: DESIGN REVIEW FIXES (2 sprints)
# ============================================================================

# FIX 5: Socket Constraint Audit Logging
# ============================================================================
# Add after line 2120 (dispatch_fig chart) in tabs[4]:

def audit_socket_constraints(result_df: pd.DataFrame) -> str:
    """
    Audit whether FCR-N and aFRR bids respect shared resource socket.
    Returns audit message for display.
    """
    if result_df.empty:
        return "No data to audit"
    
    audit_rows = []
    
    for resource in ["bess", "ev", "hvac"]:
        fcr_col = f"{resource}_fcr_kw"
        afrr_up_col = f"{resource}_afrr_up_kw"
        afrr_down_col = f"{resource}_afrr_down_kw"
        
        if not all(col in result_df.columns for col in [fcr_col, afrr_up_col, afrr_down_col]):
            continue
        
        # Check if bids imply additive socket (WRONG)
        # Correct assumption: max(fcr, afrr_up, afrr_down) ≤ available
        for idx in result_df.index[:5]:  # Sample first 5 intervals
            fcr_bid = float(result_df.loc[idx, fcr_col])
            afrr_up = float(result_df.loc[idx, afrr_up_col])
            afrr_down = float(result_df.loc[idx, afrr_down_col])
            
            # If all three are non-zero, flag potential socket conflict
            if fcr_bid > 10 and afrr_up > 10:
                audit_rows.append({
                    "Resource": resource,
                    "Interval": str(idx),
                    "FCR (kW)": fcr_bid,
                    "aFRR up (kW)": afrr_up,
                    "⚠️ Potential socket conflict": "YES"
                })
    
    if audit_rows:
        return (
            "⚠️ **Socket Audit Warning**: Some intervals show non-zero FCR and aFRR bids for the same resource. "
            "Verify optimizer doesn't violate shared socket constraint:\n"
            f"max(fcr_kw, afrr_up_kw, afrr_down_kw) ≤ available_kw_for_resource"
        )
    return "✓ Socket audit OK: No obvious conflicts detected"


# FIX 6: Improved Waterfall Chart Semantics (Lines 2132-2147)
# ============================================================================
# BEFORE: Misleading relative changes as if they're sequential stages
# bid_waterfall = go.Figure(go.Waterfall(
#     measure=["absolute", "relative", "relative", "relative", "absolute"],
#     x=["Raw symmetric", "Dynamic cap", "Energy cap", "Risk buffer", "Final bid"],

# AFTER: Correct semantics showing constraints, not stages
def create_semantically_correct_bid_waterfall(waterfall_row) -> go.Figure:
    """
    Create waterfall showing bid reduction as constraint layers,
    not sequential decomposition.
    """
    raw_kw = float(waterfall_row.get("raw_fcr_symmetric_kw", 0.0))
    dynamic_kw = min(raw_kw, float(waterfall_row.get("fcr_dynamic_cap_total_kw", raw_kw)))
    energy_kw = min(dynamic_kw, float(waterfall_row.get("fcr_energy_cap_total_kw", dynamic_kw)))
    buffer_kw = float(waterfall_row.get("reserve_buffer_kw", 0.0))
    final_bid_kw = float(waterfall_row.get("fcr_bid_kw", 0.0))
    
    # Corrected labels that show these are constraint layers
    bid_waterfall = go.Figure(
        go.Waterfall(
            measure=["absolute", "relative", "relative", "relative", "absolute"],
            x=[
                "Available capacity",
                "Limited by response cap",
                "Limited by endurance (1h)",
                "Reduced by risk buffer",
                "Final bid submitted"
            ],
            y=[
                raw_kw / 1000.0,
                -(raw_kw - dynamic_kw) / 1000.0,
                -(dynamic_kw - energy_kw) / 1000.0,
                -buffer_kw / 1000.0,
                final_bid_kw / 1000.0,
            ],
            connector={"line": {"dash": "solid"}},
            textposition="outside",
        )
    )
    bid_waterfall.update_layout(
        yaxis_title="MW",
        title_text="FCR-N Bid Decomposition: Constraints Applied Sequentially"
    )
    return bid_waterfall


# FIX 7: Lower MPC Timing Lock to Prevent Concurrent Solves (Lines 1928-1957)
# ============================================================================
# BEFORE: No synchronization, fragment reruns every 4s causing cache thrashing
# if lower_armed and live_market_session and market_status["status"] in {"live", "closed"}:
#     lower_result = cached_lower_mpc_from_upper(...)
#     st.session_state["live_lower_result"] = lower_result

# AFTER: Use session flag to prevent concurrent solves
def render_live_market_tile_with_timing_lock() -> bool:
    """
    Render live market tile with lower MPC timing lock to prevent
    concurrent solves or cache thrashing.
    """
    live_market_session = st.session_state.get("live_market_session", {})
    market_status = live_market_status(live_market_session)
    lower_armed = bool(st.session_state.get("lower_mpc_armed", False))
    
    # ... existing code ...
    
    # NEW: Implement lower MPC solving lock
    if lower_armed and live_market_session and market_status["status"] in {"live", "closed"}:
        # Check if we're already solving
        if st.session_state.get("_lower_mpc_solving", False):
            st.warning("⏳ Lower MPC is still solving from previous tick. Waiting...")
            return False
        
        elapsed_tick = int(float(market_status["elapsed_seconds"]) // 4) + 1
        prev_tick = st.session_state.get("_lower_mpc_last_tick", -1)
        
        # Only solve if time actually advanced
        if elapsed_tick > prev_tick:
            try:
                st.session_state["_lower_mpc_solving"] = True
                lower_result = cached_lower_mpc_from_upper(
                    df=df,
                    upper_result=upper_socket,
                    fleet_meta=fleet_meta,
                    market_mode=str(live_market_session.get("market_mode", market_mode)),
                    resource_mode=str(live_market_session.get("resource_mode", resource_mode)),
                    preview_4s=runtime_market_feed,
                    device_roster=device_roster,
                    inner_dt_seconds=4,
                    elapsed_tick=elapsed_tick,
                )
                st.session_state["live_lower_result"] = lower_result
                st.session_state["_lower_mpc_last_tick"] = elapsed_tick
                st.session_state["mpc_phase"] = "lower_live"
            finally:
                st.session_state["_lower_mpc_solving"] = False
    
    return True


# ============================================================================
# PRIORITY 3: POLISH FIXES (Nice to have)
# ============================================================================

# FIX 8: Ensure aFRR Energy Audit Populated or Labeled (Lines 2395-2417)
# ============================================================================
# AFTER line 2395:
# afrr_energy_audit = result_frame(lower_view_result, "afrr_energy_audit")
# if afrr_energy_audit.empty:
#     afrr_energy_audit = result_frame(result, "afrr_energy_audit")

# IMPROVEMENT: Add clear labeling
# if not afrr_energy_audit.empty:
#     st.markdown("### aFRR settlement-period energy audit")
#     # ... existing audit code ...
# else:
#     # NEW: Clear indication that audit is pending
#     audit_status = "✓ Complete" if lower_view_result.get("compliance", {}).get("afrr_energy_reporting_evaluated") else "⏳ Pending"
#     st.info(
#         f"🔔 aFRR Energy Audit: {audit_status}\n\n"
#         f"The energy audit requires the lower MPC to complete a full settlement period (15 minutes). "
#         f"Once available, it will show reconciliation between BSP-reported and Fingrid-calculated aFRR energy."
#     )


# FIX 9: Normalize Heatmaps to Consistent Resolution (Line 978)
# ============================================================================
# BEFORE:
# def availability_heatmap(df: pd.DataFrame, column: str, title: str, color_scale: str) -> go.Figure:
#     heat = df.copy()
#     heat["day"] = heat.index.strftime("%Y-%m-%d")
#     heat["hour"] = heat.index.hour
#     pivot = heat.pivot_table(index="day", columns="hour", values=column, aggfunc="mean")

# AFTER: Normalize to 15-minute resolution
def availability_heatmap_normalized(df: pd.DataFrame, column: str, title: str, color_scale: str) -> go.Figure:
    """
    Create heatmap with normalized resolution (15-min) for consistency across
    4-second and 15-minute data sources.
    """
    heat = df.copy()
    
    # Normalize resolution: resample high-resolution data to 15-minute
    try:
        freq = pd.infer_freq(heat.index)
        if freq and freq.startswith('4S'):  # 4-second data
            heat = heat.resample('15T').mean()
    except:
        pass  # If resampling fails, continue with original
    
    heat["day"] = heat.index.strftime("%Y-%m-%d")
    heat["hour"] = heat.index.hour
    pivot = heat.pivot_table(index="day", columns="hour", values=column, aggfunc="mean")
    
    fig = go.Figure(
        go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=color_scale, colorbar={"title": column})
    )
    fig.update_layout(height=320, title=title, margin=dict(l=20, r=20, t=45, b=15))
    fig.update_traces(colorbar={"title": {"text": column, "font": {"size": 16}}, "tickfont": {"size": 14}})
    return fig


# FIX 10: Fix Response Slider Labels (Lines 1411-1418)
# ============================================================================
# BEFORE: Misleading labels suggesting these affect the optimizer
# tau_bess_s = c1.slider("Visual-only BESS time constant (s)", ...)

# AFTER: Clearer labeling
# if dev_mode:
#     c1, c2, c3, c4 = st.columns(4)
#     tau_bess_s = c1.slider(
#         "[CHART ONLY] BESS τ (s)",
#         1.0, 5.0, 2.0, 0.5,
#         help="Visualization only. Optimizer uses fixed device classes, not these sliders."
#     )
#     tau_ev_s = c2.slider(
#         "[CHART ONLY] EV τ (s)",
#         1.0, 8.0, 4.0, 0.5,
#         help="Visualization only. Optimizer uses fixed device classes, not these sliders."
#     )
#     # ... etc ...
#     
#     st.warning(
#         "💡 **Response model sliders are visualization-only diagnostic tools.** "
#         "To change actual response dynamics used by the optimizer, "
#         "modify the 'Delivered HVAC response time' in the Scenario Setup tab, "
#         "or train custom forecasters in the Forecasting Lab."
#     )
# else:
#     st.info(
#         "Response model parameters are fixed per scenario. "
#         "Switch to Developer mode to visualize different response profiles."
#     )
