# Production VPP: 4-Second Real-Time Controller Architecture

## Quick Answer

**YES, you absolutely need a real-time 4-second MPC (or a faster reactive controller).** Your current Residential VPP simulator runs 4-second tracking **offline** (after the fact). Real VPPs need **online real-time control**.

---

## The Critical Problem Your Simulation DOESN'T Handle

Your current code flow:

```
Hour H: 
  1. 15-min MPC decides: "Commit 100 kW BESS, 50 kW EV, 20 kW HVAC"
  2. Send bid to Fingrid
  3. (LATER) Simulate 4-second tracking offline
  
But in REALITY:
  1. 15-min MPC decides: "Commit 100 kW BESS, 50 kW EV, 20 kW HVAC"
  2. Send bid to Fingrid
  3. Fingrid sends 4-second control signal RIGHT NOW
  4. ← WHO DECIDES HOW TO SPLIT THAT SIGNAL ACROSS BESS/EV/HVAC IN 4 SECONDS?
  5. ← WHAT IF EV DISCONNECTS IN 3 SECONDS?
```

**Answer**: A **real-time 4-second controller** must run continuously, making split-second decisions.

---

## Architecture: Hourly MPC + 4-Second Controller (Two-Layer)

```
┌─────────────────────────────────────────────────────────────────┐
│ PRODUCTION VPP ARCHITECTURE (Real Implementation)              │
└─────────────────────────────────────────────────────────────────┘

LAYER 1: HOURLY PLANNER (15-min MPC in your code)
┌──────────────────────────────────────────────────┐
│ Runs every 15 minutes                           │
│                                                  │
│ Input: Prices, forecasts, weather, availability │
│                                                  │
│ Decision:                                        │
│  "Next 15 min: Bid 100kW for FCR-N"            │
│  "Allocate: BESS=50%, EV=40%, HVAC=10%"        │
│  (based on energy constraints & response times) │
│                                                  │
│ Output: COMMITMENT SIGNAL sent to Fingrid      │
│ (This is your run_mpc_controller output)        │
└──────────────────────────────────────────────────┘
              ↓ (updates every 15 min)
              
LAYER 2: 4-SECOND REAL-TIME CONTROLLER ← YOU NEED THIS!
┌──────────────────────────────────────────────────┐
│ Runs every 4 seconds (continuous loop)          │
│                                                  │
│ Input:                                           │
│  - Fingrid's 4-second control signal            │
│  - Current state of all appliances              │
│    (BESS SOC, EV connected?, HVAC temp)        │
│  - Allocation plan from Layer 1                 │
│                                                  │
│ DECISION (every 4 seconds):                     │
│  1. Read: Fingrid says "Provide 75 kW up"     │
│  2. Check: BESS available? EV still connected? │
│  3. Allocate: 40 kW from BESS, 25 kW from EV  │
│  4. Send: Commands to each device simultaneously│
│     BESS: "ramp to 40 kW discharge NOW"       │
│     EV: "ramp to 25 kW discharge NOW"         │
│     HVAC: "maintain baseline (0 kW deviation)" │
│  5. Verify: Did devices respond? Measure power│
│  6. Adjust: If EV dropped out, compensate:    │
│     "BESS, increase to 65 kW in next 4s"      │
│                                                  │
│ Output: Actual power delivered to grid         │
└──────────────────────────────────────────────────┘
              ↓ (updates every 4 seconds)
              
PHYSICAL DEVICES
┌──────────────────────────────────────────────────┐
│ ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│ │  BESS    │  │   EV     │  │  HVAC    │       │
│ │ 5 kW/10h │  │ 7 kW/60kWh│ │ 4.6 kW  │       │
│ │ SOC=75%  │  │ SOC=40%  │  │ Temp=20°C        │
│ │ Online   │  │ CONNECTED│  │ Online           │
│ └──────────┘  └──────────┘  └──────────┘       │
└──────────────────────────────────────────────────┘
```

---

## Real-Time 4-Second Controller: What It Does

### 1. **State Monitoring (Every 4 seconds)**

```python
class RealTimeController:
    def __init__(self, devices: List[Device]):
        self.devices = devices  # [BESS, EV, HVAC]
        self.state = {}
        
    def read_device_state(self) -> Dict:
        """Poll all devices RIGHT NOW"""
        state = {
            "bess": {
                "soc_mwh": self.devices["bess"].read_soc(),        # Via Modbus/CAN
                "available_up_kw": 5.0,  # Can discharge 5 kW max
                "available_down_kw": 4.8,  # Can charge 4.8 kW max
                "response_time_s": 2.0,  # Confirmed spec
                "online": self.devices["bess"].is_online(),  # Heartbeat check
            },
            "ev": {
                "soc_mwh": self.devices["ev"].read_soc(),
                "connected": self.devices["ev"].is_plugged_in(),  # ← CRITICAL!
                "available_up_kw": 4.2,  # Can discharge if V2G enabled
                "available_down_kw": 6.8,  # Can charge
                "response_time_s": 4.0,
                "online": self.devices["ev"].is_online(),
                "departure_time_minutes": self.devices["ev"].read_departure_soc(),
            },
            "hvac": {
                "indoor_temp_c": self.devices["hvac"].read_temp(),
                "available_up_kw": 1.2,  # Limited by comfort band
                "available_down_kw": 0.8,
                "response_time_s": 180.0,  # SLOW!
                "online": self.devices["hvac"].is_online(),
            },
            "grid_frequency_hz": self.read_frequency(),  # From meter
            "fingrid_signal_norm": self.read_fingrid_signal(),  # From API/RTU
        }
        return state
    
    def execute_control_cycle(self, 
                             fingrid_request_kw: float,
                             hourly_plan: Dict) -> Dict:
        """This runs EVERY 4 SECONDS in production"""
        
        # Step 1: Measure RIGHT NOW
        state = self.read_device_state()
        
        # Step 2: Check for failures/disconnections
        if not state["ev"]["connected"]:
            print("WARNING: EV disconnected! Reallocate from 50 kW → other devices")
            hourly_plan["ev_allocation_kw"] = 0  # Zero out EV
        
        if state["bess"]["soc_mwh"] < 0.5:  # Almost empty
            print("CRITICAL: BESS near empty, reduce discharge request")
            hourly_plan["bess_allocation_kw"] *= 0.5
        
        # Step 3: Allocate power request
        #         This is WHERE THE 4-SEC MPC LOGIC GOES
        
        allocation = self.allocate_power(
            fingrid_request_kw=fingrid_request_kw,
            available_devices=state,
            preferred_allocation=hourly_plan,  # From 15-min MPC
            max_response_time_s=4.0  # Real-time constraint
        )
        
        # Step 4: Send commands to devices simultaneously
        self.send_setpoint(device="bess", power_kw=allocation["bess_kw"])
        self.send_setpoint(device="ev", power_kw=allocation["ev_kw"])
        self.send_setpoint(device="hvac", power_kw=allocation["hvac_kw"])
        
        # Step 5: Verify execution (after ~1 second)
        time.sleep(1.0)
        actual_state = self.read_device_state()
        actual_power = (
            actual_state["bess"]["actual_power_kw"] +
            actual_state["ev"]["actual_power_kw"] +
            actual_state["hvac"]["actual_power_kw"]
        )
        
        # Step 6: Diagnostic logging
        return {
            "timestamp": now,
            "fingrid_request_kw": fingrid_request_kw,
            "requested_allocation": allocation,
            "actual_delivered_kw": actual_power,
            "accuracy_ratio": actual_power / max(fingrid_request_kw, 0.1),
            "errors": [...]
        }
```

### 2. **Power Allocation Logic (The Decision Maker)**

```python
def allocate_power(self,
                   fingrid_request_kw: float,
                   available_devices: Dict,
                   preferred_allocation: Dict,
                   max_response_time_s: float = 4.0) -> Dict:
    """
    WHO DECIDES WHICH DEVICE PROVIDES HOW MUCH?
    
    This is a real-time optimization problem (4-second MPC).
    It must:
    1. Respect device constraints (SOC, response time, comfort)
    2. Follow Fingrid's request ±5% accuracy
    3. React to disconnections
    4. Prefer fast devices (BESS first)
    """
    
    # Read request
    request_power = fingrid_request_kw  # e.g., +75 kW (discharge)
    
    # Filter: Only use devices with response < 4s
    fast_devices = [
        d for d in self.devices
        if available_devices[d]["response_time_s"] <= max_response_time_s
    ]
    # Result: [BESS (2s), EV (4s)] — HVAC (180s) excluded for fast response
    
    # Priority allocation: Fastest first (greedy)
    allocation = {}
    remaining_request = request_power
    
    # 1st Priority: BESS (fastest, most controllable)
    bess_max = available_devices["bess"]["available_up_kw"]
    bess_alloc = min(remaining_request, bess_max)
    allocation["bess_kw"] = bess_alloc
    remaining_request -= bess_alloc
    
    # 2nd Priority: EV (medium speed)
    if remaining_request > 0 and available_devices["ev"]["connected"]:
        ev_max = available_devices["ev"]["available_up_kw"]
        ev_alloc = min(remaining_request, ev_max)
        allocation["ev_kw"] = ev_alloc
        remaining_request -= ev_alloc
    else:
        allocation["ev_kw"] = 0  # EV not available
    
    # 3rd Priority: HVAC (slow, use only inside its deliverable cap)
    #              Only if we still need power AND the product audit allows the delay
    if remaining_request > 0 and max_response_time_s >= 300:  # aFRR audits full activation over 5 minutes
        hvac_max = available_devices["hvac"]["available_up_kw"]
        hvac_alloc = min(remaining_request, hvac_max)
        allocation["hvac_kw"] = hvac_alloc
        remaining_request -= hvac_alloc
    else:
        allocation["hvac_kw"] = 0
    
    # Check: Can we fulfill the request?
    total_allocated = (allocation["bess_kw"] + 
                       allocation["ev_kw"] + 
                       allocation["hvac_kw"])
    
    if total_allocated < request_power * 0.95:  # Less than 95% accuracy
        print(f"WARNING: Insufficient capacity!")
        print(f"  Requested: {request_power} kW")
        print(f"  Can deliver: {total_allocated} kW")
        print(f"  Shortfall: {request_power - total_allocated} kW")
        # → Fingrid will see lower frequency response, impact revenue
    
    return allocation
```

### 3. **Handling Device Disconnection**

```python
# Time: T = 0s
# MPC decided: BESS 50%, EV 40%, HVAC 10% of 100 kW commitment
# → BESS=50kW, EV=40kW, HVAC=10kW

# Time: T = 4s
# Fingrid signal: "Provide 75 kW up"
# Real-time controller allocates:
#   BESS: 40 kW (within limits)
#   EV: 35 kW (within limits)  
#   HVAC: 0 kW (too slow for tight margin)
# → Total: 75 kW delivered ✓

# Time: T = 8s
# Fingrid signal: "Provide 70 kW up"
# Status: EV DISCONNECTS (driver unplugged car!)
# Real-time controller detects:
#   EV state = {"connected": false}
# → REALLOCATE:
#   Was: BESS=40 kW, EV=30 kW (from preferred allocation)
#   Now: BESS=50 kW (increase!), EV=0 kW, HVAC=20 kW
# → Total: 70 kW delivered ✓

# But if EV disconnection happens with heavy load:
# Time: T = 12s
# Fingrid signal: "Provide 90 kW up"
# EV STILL disconnected
# Real-time controller tries:
#   BESS: 50 kW max (at limit, hitting SOC constraints)
#   EV: 0 kW (disconnected)
#   HVAC: 30 kW (emergency mobilization)
# → Total: 80 kW (SHORTFALL: 10 kW!)
#
# Result: ACCURACY FAILS
#   - Revenue penalized: miss 10 kW × energy price
#   - Compliance risk: reserve audit expects requested delivery inside tolerance
#   - Next hour: Either reduce bid or add HVAC to fleet
```

---

## The MPC at 4-Second Timescale

Your 15-minute MPC **cannot** run every 4 seconds (too slow, too expensive):

```
Timescale     | Frequency | Purpose              | Horizon   | CPU
──────────────┼───────────┼──────────────────────┼───────────┼──────
15 min MPC    | Every 15m | Plan next 60-120 min | 2-4 hours | High
4 sec Control | Every 4s  | React to signal NOW  | 4-16 sec  | Low
```

**Instead, you need a REACTIVE controller** (simpler, faster):

```python
class FastReactiveController:
    """Runs every 4 seconds, NO optimization, pure greedy allocation"""
    
    def __init__(self, hourly_plan: Dict):
        self.hourly_plan = hourly_plan  # Guideline from MPC
        
    def react(self, fingrid_signal_kw: float, device_state: Dict) -> Dict:
        """Execute in < 100ms"""
        
        # Option A: Greedy (priority allocation)
        # Allocate to fastest devices first
        allocation = self.greedy_allocate(
            request=fingrid_signal_kw,
            devices=device_state
        )
        
        # Option B: Fuzzy logic (simple rules-based)
        # "If frequency < 49.9 Hz and BESS > 30%, discharge BESS hard"
        allocation = self.fuzzy_allocate(
            frequency_hz=device_state["frequency"],
            bess_soc=device_state["bess_soc"],
            ev_connected=device_state["ev_connected"]
        )
        
        # Option C: Hysteresis (avoid oscillation)
        # "Keep current allocation unless state changes significantly"
        allocation = self.hysteresis_allocate(
            new_signal=fingrid_signal_kw,
            previous_signal=self.prev_signal,
            device_state=device_state
        )
        
        return allocation
```

---

## Real Production Example: Sonnen EnergiePro

**How does Sonnen's VPP actually work?**

```
LAYER 1: Sonnen Server (Cloud)
  - 15-min MPC optimization
  - Aggregates 10,000+ home batteries
  - Predicts next 24 hours
  - Sends: "Your profile for next 15 min: 
            Charge at 2 kW, reserve 5 kW for FCR-N"

LAYER 2: Sonnen Battery (Home)
  - Receives profile from cloud
  - Runs 4-second real-time loop locally
  - Monitors grid frequency via meter
  - When frequency dips:
    → Automatically discharge to support grid
    → No cloud latency!
  - Reports back to cloud every minute:
    "Delivered 4.5 kW FCR energy, 
     SOC now 65%, battery healthy"

LAYER 3: National TSO (Fingrid in Finland)
  - Sends 4-second control signal to aggregator
  - "Frequency = 49.92 Hz, need -50 MW response"
  - Aggregator broadcasts to all 10,000 batteries:
    "Discharge hard!"
  - Each battery reacts independently in real-time
  - Fingrid sees combined response within 2 seconds
```

**Key insight**: Real-time control happens **locally on the device**, not in cloud!

---

## What Happens If You DON'T Have 4-Second Controller?

### Scenario: You Only Have 15-Min MPC (Like Current Residential VPP)

```
Time 0:00-0:15
  MPC: "Commit 100 kW for 15 minutes"
  (Decision made once, locked in)

Time 0:04
  Fingrid: "Grid needs 75 kW up NOW"
  Your response: ??? 
  (15-min plan can't react, too slow to replan)

Time 0:08
  EV disconnects
  Fingrid: "Grid needs 80 kW up"
  Your response: Still trying to execute old 15-min plan
  → FAILURE: Can't meet request, revenue lost

Outcome:
  ✗ Non-compliant with FCR-N dynamic-response screen
  ✗ Cannot handle device failures
  ✗ Miss revenue opportunities
  ✗ Risk penalty if accuracy < 90%
```

### Scenario: You Have 4-Second Reactive Controller (Correct Architecture)

```
Time 0:00-0:15
  MPC: "Commit 100 kW for 15 minutes"
  (Guideline sent to home)

Time 0:04
  Fingrid: "Grid needs 75 kW up NOW"
  4-sec controller: Reads signal, allocates:
    BESS 50 kW + EV 25 kW = 75 kW
  Devices respond in 2-4 seconds
  → SUCCESS: Meets request

Time 0:08
  EV disconnects
  Fingrid: "Grid needs 80 kW up"
  4-sec controller: Detects EV offline, reallocates:
    BESS 50 kW + HVAC 30 kW = 80 kW
  Devices respond
  → SUCCESS: Compensates for failure

Outcome:
  ✓ Tracks inside the simulated reserve socket
  ✓ Handles device failures gracefully
  ✓ Captures all revenue
  ✓ Accuracy > 95%
```

---

## Implementation: What You'd Build for Production

### Software Architecture

```
┌─────────────────────────────────────┐
│ Cloud (Hourly Optimization MPC)     │
│ - Run 15-min MPC every 15 minutes   │
│ - Forecast prices, optimize revenue │
│ - Send profile to home              │
│                                      │
│ Python/MATLAB → Linear Programming  │
│ CPU: Can afford 10 minutes compute  │
└─────────────────────────────────────┘
              ↓ (every 15 min via API)
              
┌─────────────────────────────────────┐
│ Home Controller (Real-Time Loop)    │
│                                     │
│ Runs every 4 seconds:               │
│ 1. Read grid frequency/signal       │
│ 2. Check device status              │
│ 3. Allocate power (greedy rules)   │
│ 4. Send commands to devices         │
│ 5. Measure response                 │
│ 6. Log for compliance               │
│                                     │
│ Python/C → Simple logic (< 50ms)   │
│ CPU: Raspberry Pi or embedded       │
└─────────────────────────────────────┘
         ↓ (every 4 seconds)
         
┌─────────────────────────────────────┐
│ Devices (Hardware Layer)            │
│                                     │
│ BESS: Modbus/CAN bus               │
│ EV: ISO 15118 V2G protocol         │
│ HVAC: WiFi thermostat              │
│                                     │
│ Each has local failsafe             │
│ (disconnect if no signal for 10s)  │
└─────────────────────────────────────┘
```

### Communication Protocols

```
Home → Grid:
  - 4G/5G or fiber → Report power, frequency
  - 4G/5G ← Receive control signal every 4s
  
Home → BESS:
  - Modbus RTU over RS-485 (industrial standard)
  - "Set power to 50 kW discharge"
  
Home → EV:
  - ISO 15118 (V2G standard)
  - "Discharge at 6.6 kW, stop in 5 minutes"
  
Home → HVAC:
  - WiFi/Zigbee smart thermostat
  - "Increase setpoint by 1°C"
```

---

## Answer to Your Questions

### Q1: "Who decides which appliance provides how much?"

**A**: The **4-second real-time controller** on your home hardware.

Algorithm:
1. **Greedy allocation**: BESS first (fastest), then EV, then HVAC
2. **Constraint checking**: Don't exceed device limits, SOC bounds, comfort band
3. **Failure handling**: If a device goes offline, rebalance across remaining devices
4. **Guidance from 15-min MPC**: "Prefer 50% BESS, 40% EV, 10% HVAC" but adapt in real-time

**Code location in production**:
```python
# This function runs every 4 seconds locally on your home controller:
def allocate_power_realtime(fingrid_request, device_state, hourly_guideline):
    # Allocation logic here (greedy or fuzzy rules)
    return allocation_for_each_device
```

### Q2: "If one appliance suddenly turns off, what happens?"

**A**: The 4-second controller **detects it and reallocates immediately**.

Example:
```
Before: BESS 40 kW + EV 35 kW + HVAC 0 kW = 75 kW total
EV disconnects (driver unplugs car)
After (next 4-sec cycle): BESS 50 kW + EV 0 kW + HVAC 25 kW = 75 kW total
```

**Detection**: 
- Poll device status every 4 seconds via Modbus/WiFi
- Check heartbeat: "Is EV still responding?" 
- If no response for 8 seconds (2 cycles) → mark offline

**Reallocation**:
- Increase BESS discharge to compensate
- Mobilize HVAC if available (respects comfort band)
- If shortfall, alert cloud system for next 15-min MPC replan

**Failure scenario**:
- EV disconnect happens
- BESS at 30% SOC (can't go lower)
- HVAC limited by comfort (only 20 kW available)
- → **Shortfall**: Can only deliver 50 kW instead of requested 75 kW
- → **Impact**: Revenue reduced, accuracy metric fails
- → **Next step**: Cloud MPC reduces bid for next hour

---

## Your Residential VPP: Road to Production

### Current State (Simulation)

```python
run_mpc_controller()  # 15-min MPC
  ├─ Hourly planner ✓
  ├─ Forecasting ✓
  └─ Offline 4-sec tracking (for analysis only) ⚠
```

### Production Addition Needed

```python
class RealTimeController:  # ← ADD THIS
    """Runs every 4 seconds on home device"""
    
    def __init__(self, hourly_plan):
        self.hourly_plan = hourly_plan  # From MPC
        self.devices = [BESS, EV, HVAC]
        self.prev_allocation = {}
        
    def control_loop(self):
        while True:
            # Read signal
            fingrid_signal = self.read_grid_signal()
            device_state = self.poll_devices()
            
            # Allocate
            allocation = self.allocate_power_realtime(
                signal=fingrid_signal,
                state=device_state,
                guideline=self.hourly_plan
            )
            
            # Send
            self.send_commands(allocation)
            
            # Sleep until next cycle
            time.sleep(4.0)
```

**Integration with your MPC**:
```python
# In your Streamlit dashboard:
mpc_result = run_mpc_controller(...)  # Hourly plan
hourly_guideline = mpc_result["controls"]

# Send to home device:
send_to_home(hourly_guideline)

# Home device runs this continuously:
realtime = RealTimeController(hourly_guideline)
realtime.control_loop()  # Every 4 seconds forever
```

---

## Summary Table

| Aspect | 15-Min MPC | 4-Sec Controller |
|--------|-----------|-----------------|
| **Runs** | Every 15 minutes | Every 4 seconds (continuous) |
| **Compute** | Complex optimization (PuLP, 10-60s) | Simple rules (< 50ms) |
| **Input** | Forecasts, prices, historical data | Current grid signal, device status |
| **Output** | Bid plan (kWh per hour) | Dispatch command (kW per device) |
| **Where** | Cloud server | Home hardware (Raspberry Pi/gateway) |
| **Decision** | "What to bid?" | "How to fulfill bid RIGHT NOW?" |
| **Reacts to** | Forecast changes | Grid frequency, device failures |
| **Latency** | Can afford 10-60 seconds | Must complete in < 100ms |
| **Failure handling** | Replan next cycle | Adapt immediately within 4 seconds |

---

## References

- **Sonnen EnergiePro**: https://sonnen.de (real-world VPP example)
- **Fingrid reserve product specs**: https://www.fingrid.fi/en/electricity-market/reserves/ (FCR-N local activation, aFRR activation, and market requirements)
- **ISO 15118 V2G**: https://www.iso.org/standard/69638.html (EV charging protocol)
- **Modbus RTU**: https://modbus.org/ (battery communication)
