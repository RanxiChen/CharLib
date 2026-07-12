import PySpice
from numpy import average
from collections import defaultdict

from charlib.characterizer import utils, plots
from charlib.characterizer.cell import Port
from charlib.characterizer.procedures import register, ProcedureFailedException
from charlib.liberty import liberty
from charlib.liberty.library import LookupTable

@register('data_slews', 'loads', 'transient_sim_end_time')
def combinational_worst_case(cell, config, settings):
    for variation in config.variations('data_slews', 'loads', 'transient_sim_end_time'):
        for path in cell.paths():
            yield (measure_delays_for_path_with_criterion, cell, config, settings, variation, path, max)

@register('data_slews', 'loads', 'transient_sim_end_time')
def combinational_average(cell, config, settings):
    for variation in config.variations('data_slews', 'loads', 'transient_sim_end_time'):
        for path in cell.paths():
            yield (measure_delays_for_path_with_criterion, cell, config, settings, variation, path, average)


# ── Phase C: compact measurement (per state_map, not per path) ───────────

def _measure_one_state_map(cell, config, settings, variation, path, state_map):
    """Run ONE simulation for one state_map. Returns compact dict with VALUES IN LIBERTY UNITS."""
    [input_pin, _, output_pin, _] = path
    data_slew = variation['data_slews'] * settings.units.time
    load = variation['loads'] * settings.units.capacitance
    t_sim_end = max(variation['transient_sim_end_time'] * settings.units.time, 1000*data_slew)
    vdd = settings.primary_power.voltage * settings.units.voltage
    vss = settings.primary_ground.voltage * settings.units.voltage

    circuit = utils.init_circuit('comb_delay', cell.netlist, config.models,
                                 settings.named_nodes, settings.units)
    pin_map = utils.PinStateMap(cell.inputs, cell.outputs, state_map)
    connections = []
    measurements = []
    measurement_names = set()

    for pin in cell.pins_in_netlist_order():
        match pin.role:
            case Port.Role.LOGIC:
                if pin.name in pin_map.target_inputs:
                    connections.append(f"v{pin.name}")
                    (v_0, v_1) = (vss, vdd) if pin_map.target_inputs[pin.name] == "01" else (vdd, vss)
                    circuit.PieceWiseLinearVoltageSource(
                        pin.name, f"v{pin.name}", circuit.gnd,
                        values=utils.slew_pwl(v_0, v_1, data_slew, 3*data_slew,
                                              settings.logic_thresholds.low,
                                              settings.logic_thresholds.high))
                elif pin.name in pin_map.target_outputs:
                    connections.append(f"v{pin.name}")
                    circuit.C(pin.name, f"v{pin.name}", circuit.gnd, load)
                    for in_pin in pin_map.target_inputs:
                        if pin_map.target_inputs[in_pin] == "01":
                            in_direction = "rise"
                            threshold_prop_0 = settings.logic_thresholds.rising
                        else:
                            in_direction = "fall"
                            threshold_prop_0 = settings.logic_thresholds.falling
                        if pin_map.target_outputs[pin.name] == "01":
                            out_direction = "rise"
                            threshold_prop_1 = settings.logic_thresholds.rising
                            threshold_tran_0 = settings.logic_thresholds.low
                            threshold_tran_1 = settings.logic_thresholds.high
                        else:
                            out_direction = "fall"
                            threshold_prop_1 = settings.logic_thresholds.falling
                            threshold_tran_0 = settings.logic_thresholds.high
                            threshold_tran_1 = settings.logic_thresholds.low
                        prop_name = f"cell_{out_direction}__{in_pin}_to_{pin.name}".lower()
                        measurement_names.add(prop_name)
                        measurements.append((
                            "tran", prop_name,
                            f"trig v(v{in_pin}) val={float(vdd*threshold_prop_0)} {in_direction}=1",
                            f"targ v(v{pin.name}) val={float(vdd*threshold_prop_1)} {out_direction}=1"))
                        tran_name = f"{out_direction}_transition__{in_pin}_to_{pin.name}".lower()
                        measurement_names.add(tran_name)
                        measurements.append((
                            "tran", tran_name,
                            f"trig v(v{pin.name}) val={float(vdd*threshold_tran_0)} {out_direction}=1",
                            f"targ v(v{pin.name}) val={float(vdd*threshold_tran_1)} {out_direction}=1"))
                elif pin.name in pin_map.stable_inputs:
                    connections.append(settings.primary_ground.name if pin_map.stable_inputs[pin.name] == "0"
                                       else settings.primary_power.name)
                elif pin.name in pin_map.ignored_outputs:
                    connections.append("wfloat0")
                else:
                    raise ValueError(f"Unable to connect unrecognized logic pin {pin.name}")
            case Port.Role.POWER:
                connections.append(settings.primary_power.name)
            case Port.Role.GROUND:
                connections.append(settings.primary_ground.name)
            case Port.Role.NWELL:
                connections.append(settings.nwell.name)
            case Port.Role.PWELL:
                connections.append(settings.pwell.name)
            case _:
                raise ValueError(f"Unable to connect unrecognized pin {pin.name}")

    circuit.X("dut", cell.name, *connections)

    simulator = PySpice.Simulator.factory(simulator=settings.simulation.backend)
    simulation = simulator.simulation(circuit, temperature=settings.temperature,
                                      nominal_temperature=settings.temperature)
    simulation.options("autostop", "nopage", "nomod", post=1, ingold=2, trtol=1)
    for measure in measurements:
        simulation.measure(*measure, run=False)
    simulation.transient(step_time=data_slew/8, end_time=t_sim_end, run=False)

    stable_key = ",".join(["=".join([pin, state]) for pin, state in pin_map.stable_inputs.items()])
    try:
        analysis = simulator.run(simulation)
    except Exception as e:
        msg = f"Procedure failed for cell {cell.name}"
        if settings.debug:
            debug_path = settings.debug_dir / cell.name / __name__.split(".")[-1]
            debug_path.mkdir(parents=True, exist_ok=True)
        raise ProcedureFailedException(msg) from e

    # Convert to Liberty units (EXACTLY matching original per-point logic)
    time_unit = settings.units.time.prefixed_unit
    cap_unit = settings.units.capacitance.prefixed_unit
    meas_converted = {}
    for name in measurement_names:
        if name in analysis.measurements:
            delay_val = (analysis.measurements[name] @ PySpice.Unit.u_s).convert(time_unit).value
            meas_converted[name] = delay_val
        else:
            meas_converted[name] = None

    return {
        "input_pin": input_pin,
        "output_pin": output_pin,
        "slew_idx_val": data_slew.convert(time_unit).value,
        "load_idx_val": load.convert(cap_unit).value,
        "state_key": stable_key,
        "measurements": meas_converted,
    }


def batch_run_compact_delay_points(cell, config, settings, batch_points):
    results = []
    for (variation, path, state_map) in batch_points:
        try:
            result = _measure_one_state_map(cell, config, settings, variation, path, state_map)
            results.append(result)
        except ProcedureFailedException:
            if settings.omit_on_failure:
                continue
            raise
    return results


def build_liberty_from_compact_rows(cell, config, settings, compact_rows, criterion=max):
    """Build Liberty cell group from compact rows (values already in Liberty units)."""
    result = cell.liberty

    pin_groups = defaultdict(list)
    for row in compact_rows:
        key = (row["output_pin"], row["input_pin"])
        pin_groups[key].append(row)

    for (output_pin, input_pin), rows in pin_groups.items():
        timing_id = f"/* {input_pin} */"
        result.group("pin", output_pin).add_group("timing", timing_id)
        result.group("pin", output_pin).group("timing", timing_id).add_attribute("related_pin", input_pin)

        all_meas_names = set()
        for row in rows:
            all_meas_names.update(row["measurements"].keys())

        for meas_name in sorted(all_meas_names):
            lut_name, _ = meas_name.split("__")
            lut_template_size = f"{len(config.parameters['loads'])}x{len(config.parameters['data_slews'])}"

            # Collect values by slew/load index
            slew_vals = sorted(set(r["slew_idx_val"] for r in rows))
            load_vals = sorted(set(r["load_idx_val"] for r in rows))
            value_map = defaultdict(list)
            for row in rows:
                val = row["measurements"].get(meas_name)
                if val is not None:
                    si = slew_vals.index(row["slew_idx_val"])
                    li = load_vals.index(row["load_idx_val"])
                    value_map[(li, si)].append(val)

            lut = LookupTable(lut_name, f"delay_template_{lut_template_size}",
                              total_output_net_capacitance=load_vals,
                              input_net_transition=slew_vals)
            for (li, si), vals in value_map.items():
                lut.values[(li, si)] = criterion(vals)
            result.group("pin", output_pin).group("timing", timing_id).add_group(lut)

    return result


# ── Original function (unchanged, used by non-batch path) ────────────────

def measure_delays_for_path_with_criterion(cell, config, settings, variation, path, criterion=max):
    [input_pin, _, output_pin, _] = path
    data_slew = variation["data_slews"] * settings.units.time
    load = variation["loads"] * settings.units.capacitance
    t_sim_end = max(variation["transient_sim_end_time"] * settings.units.time, 1000*data_slew)
    vdd = settings.primary_power.voltage * settings.units.voltage
    vss = settings.primary_ground.voltage * settings.units.voltage

    analyses = {}
    measurement_names = set()
    for state_map in cell.nonmasking_conditions_for_path(*path):
        circuit = utils.init_circuit("comb_delay", cell.netlist, config.models,
                                     settings.named_nodes, settings.units)
        pin_map = utils.PinStateMap(cell.inputs, cell.outputs, state_map)
        connections = []
        measurements = []
        for pin in cell.pins_in_netlist_order():
            match pin.role:
                case Port.Role.LOGIC:
                    if pin.name in pin_map.target_inputs:
                        connections.append(f"v{pin.name}")
                        (v_0, v_1) = (vss, vdd) if pin_map.target_inputs[pin.name] == "01" else (vdd, vss)
                        circuit.PieceWiseLinearVoltageSource(
                            pin.name, f"v{pin.name}", circuit.gnd,
                            values=utils.slew_pwl(v_0, v_1, data_slew, 3*data_slew,
                                                  settings.logic_thresholds.low,
                                                  settings.logic_thresholds.high))
                    elif pin.name in pin_map.target_outputs:
                        connections.append(f"v{pin.name}")
                        circuit.C(pin.name, f"v{pin.name}", circuit.gnd, load)
                        for in_pin in pin_map.target_inputs:
                            if pin_map.target_inputs[in_pin] == "01":
                                in_direction = "rise"
                                threshold_prop_0 = settings.logic_thresholds.rising
                            else:
                                in_direction = "fall"
                                threshold_prop_0 = settings.logic_thresholds.falling
                            if pin_map.target_outputs[pin.name] == "01":
                                out_direction = "rise"
                                threshold_prop_1 = settings.logic_thresholds.rising
                                threshold_tran_0 = settings.logic_thresholds.low
                                threshold_tran_1 = settings.logic_thresholds.high
                            else:
                                out_direction = "fall"
                                threshold_prop_1 = settings.logic_thresholds.falling
                                threshold_tran_0 = settings.logic_thresholds.high
                                threshold_tran_1 = settings.logic_thresholds.low
                            prop_name = f"cell_{out_direction}__{in_pin}_to_{pin.name}".lower()
                            measurement_names.add(prop_name)
                            measurements.append((
                                "tran", prop_name,
                                f"trig v(v{in_pin}) val={float(vdd*threshold_prop_0)} {in_direction}=1",
                                f"targ v(v{pin.name}) val={float(vdd*threshold_prop_1)} {out_direction}=1"))
                            tran_name = f"{out_direction}_transition__{in_pin}_to_{pin.name}".lower()
                            measurement_names.add(tran_name)
                            measurements.append((
                                "tran", tran_name,
                                f"trig v(v{pin.name}) val={float(vdd*threshold_tran_0)} {out_direction}=1",
                                f"targ v(v{pin.name}) val={float(vdd*threshold_tran_1)} {out_direction}=1"))
                    elif pin.name in pin_map.stable_inputs:
                        connections.append(settings.primary_ground.name if pin_map.stable_inputs[pin.name] == "0"
                                           else settings.primary_power.name)
                    elif pin.name in pin_map.ignored_outputs:
                        connections.append("wfloat0")
                    else:
                        raise ValueError(f"Unable to connect unrecognized logic pin {pin.name}")
                case Port.Role.POWER:
                    connections.append(settings.primary_power.name)
                case Port.Role.GROUND:
                    connections.append(settings.primary_ground.name)
                case Port.Role.NWELL:
                    connections.append(settings.nwell.name)
                case Port.Role.PWELL:
                    connections.append(settings.pwell.name)
                case _:
                    raise ValueError(f"Unable to connect unrecognized pin {pin.name}")

        circuit.X("dut", cell.name, *connections)

        simulator = PySpice.Simulator.factory(simulator=settings.simulation.backend)
        simulation = simulator.simulation(circuit, temperature=settings.temperature,
                                          nominal_temperature=settings.temperature)
        simulation.options("autostop", "nopage", "nomod", post=1, ingold=2, trtol=1)
        for measure in measurements:
            simulation.measure(*measure, run=False)
        simulation.transient(step_time=data_slew/8, end_time=t_sim_end, run=False)

        stable_str = ",".join(["=".join([pin, state]) for pin, state in pin_map.stable_inputs.items()])
        try:
            analyses[stable_str] = simulator.run(simulation)
        except Exception as e:
            msg = f"Procedure measure_worst_case_delay_for_path failed for cell {cell.name}"
            if settings.debug:
                debug_path = settings.debug_dir / cell.name / __name__.split(".")[-1]
                debug_path.mkdir(parents=True, exist_ok=True)
            raise ProcedureFailedException(msg) from e

    result = cell.liberty
    result.group("pin", output_pin).add_group("timing", f"/* {input_pin} */")
    result.group("pin", output_pin).group("timing", f"/* {input_pin} */").add_attribute("related_pin", input_pin)
    for name in measurement_names:
        if "io" in config.plots:
            fig = plots.plot_io_voltages(analyses.values(), list(pin_map.target_inputs.keys()),
                                         list(pin_map.target_outputs.keys()),
                                         legend_labels=analyses.keys(),
                                         indicate_voltages=[settings.primary_power.voltage*settings.logic_thresholds.low,
                                                            settings.primary_power.voltage*settings.logic_thresholds.high])
            fig_path = settings.plots_dir / cell.name / "io"
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{name}_slew_{data_slew}_load_{load}.png")
            import matplotlib.pyplot as plt
            plt.close(fig)

        delay_measurements = [analysis.measurements[name] for analysis in analyses.values() if name in analysis.measurements]
        delay = criterion(delay_measurements) @ PySpice.Unit.u_s
        lut_name, _ = name.split("__")
        lut_template_size = f"{len(config.parameters['loads'])}x{len(config.parameters['data_slews'])}"
        lut = LookupTable(lut_name, f"delay_template_{lut_template_size}",
                          total_output_net_capacitance=[load.convert(settings.units.capacitance.prefixed_unit).value],
                          input_net_transition=[data_slew.convert(settings.units.time.prefixed_unit).value])
        lut.values[0,0] = delay.convert(settings.units.time.prefixed_unit).value
        result.group("pin", output_pin).group("timing", f"/* {input_pin} */").add_group(lut)

    return result
