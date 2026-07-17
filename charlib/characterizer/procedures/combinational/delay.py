import PySpice
import matplotlib.pyplot as plt
import hashlib
import time
from numpy import average

from charlib.characterizer import utils, plots
from charlib.characterizer.cell import Port
from charlib.characterizer.procedures import register, ProcedureFailedException
from charlib.liberty import liberty
from charlib.liberty.library import LookupTable


def plan_points(cell, config, settings, variation, path, criterion):
    """Generate SimulationPoints for one (variation, path, criterion) combination."""
    from charlib.characterizer.reusable_engine import SimulationPoint

    points = []
    data_slew = float(variation['data_slews'] * settings.units.time)
    load = float(variation['loads'] * settings.units.capacitance)
    t_sim_end = float(variation['transient_sim_end_time'] * settings.units.time)
    input_pin = path[0]
    output_pin = path[1]

    for pin_states in cell.nonmasking_conditions_for_path(*path):
        pin_map = utils.PinStateMap(cell.inputs, cell.outputs, pin_states)
        state_condition = tuple(sorted((k, str(v)) for k, v in pin_states.items()))
        stable_inputs = tuple(sorted(pin_map.stable_inputs.keys()))
        ignored_outputs = tuple(sorted(pin_map.ignored_outputs))
        input_transition = pin_map.target_inputs.get(input_pin, '01')
        output_transition = pin_map.target_outputs.get(output_pin, '10')

        # Deterministic point_id from cell, path, condition, slew, load
        id_src = f"{cell.name}_{input_pin}_{output_pin}_{input_transition}_{output_transition}_{data_slew:.6e}_{load:.6e}"
        id_hash = hashlib.md5((id_src + str(state_condition)).encode()).hexdigest()[:8]
        point_id = f"{id_src}_{id_hash}"

        points.append(SimulationPoint(
            point_id=point_id,
            task_id=len(points),
            cell_name=cell.name,
            netlist_path=str(cell.netlist),
            model_paths=tuple((str(m[0]), m[1] if len(m) > 1 else '') for m in config.models),
            pin_order=tuple(p.name for p in cell.pins_in_netlist_order()),
            input_pin=input_pin,
            output_pin=output_pin,
            input_transition=input_transition,
            output_transition=output_transition,
            stable_inputs=stable_inputs,
            ignored_outputs=ignored_outputs,
            data_slew=data_slew,
            load=load,
            t_sim_end=t_sim_end,
            temperature=float(settings.temperature),
            supplies=tuple((s.name, float(s.voltage)) for s in settings.named_nodes),
            thresholds=(float(settings.logic_thresholds.low), float(settings.logic_thresholds.high)),
            time_unit_str=str(settings.units.time),
            voltage_unit_str=str(settings.units.voltage),
            criterion_group=criterion.__name__ if hasattr(criterion, '__name__') else str(criterion),
            state_condition=state_condition,
        ))
    return points


def execute_point_fresh(point, *, cell=None, config=None, settings=None, variation=None, path=None):
    """Execute one SimulationPoint with a fresh deck load.

    For R1, object kwargs (cell, config, etc.) are accepted as a legacy-oracle adapter.
    If required context is absent, raises ValueError.
    """
    from charlib.characterizer.reusable_engine import MeasurementResult, TopologySignature

    if cell is None or config is None or settings is None or variation is None or path is None:
        raise ValueError("execute_point_fresh requires legacy context objects for R1")

    t0 = time.perf_counter()
    data_slew = float(variation['data_slews'] * settings.units.time)
    load = float(variation['loads'] * settings.units.capacitance)
    t_sim_end = max(float(variation['transient_sim_end_time'] * settings.units.time), 1000*data_slew)
    vdd = settings.primary_power.voltage * settings.units.voltage
    vss = settings.primary_ground.voltage * settings.units.voltage

    # Reconstruct state_map from point.state_condition
    state_map = dict(point.state_condition)

    try:
        measurements, meas_names, pin_map, analysis, spms = _run_single_condition(
            cell, config, settings, variation, path, state_map,
            data_slew, load, t_sim_end, vdd, vss)
        t1 = time.perf_counter()

        # Build signature
        sig = TopologySignature(
            cell_hash=hashlib.md5(point.cell_name.encode()).hexdigest()[:16],
            netlist_hash=hashlib.md5(point.netlist_path.encode()).hexdigest()[:16],
            model_hashes=tuple((s or '', hashlib.md5(p.encode()).hexdigest()[:16]) for s, p in point.model_paths),
            pin_topology=tuple((p, 'input' if p in point.stable_inputs or p == point.input_pin else 'output', '')
                              for p in point.pin_order),
            state_condition=point.state_condition,
            measurement_names=tuple(sorted(meas_names)),
            measurement_directions=tuple(),
            backend=settings.simulation.backend,
            temperature=point.temperature,
            supplies_hash=hashlib.md5(str(point.supplies).encode()).hexdigest()[:16],
        )

        return MeasurementResult(
            point_id=point.point_id,
            task_id=point.task_id,
            signature=sig,
            measurement=measurements,
            status='ok',
            timings={'total': t1 - t0},
            deck_load_count=1,
        )
    except Exception as e:
        return MeasurementResult(
            point_id=point.point_id,
            task_id=point.task_id,
            status='error',
            error=str(e),
            deck_load_count=1,
        )


@register('data_slews', 'loads', 'transient_sim_end_time')
def combinational_worst_case(cell, config, settings):
    """Measure worst-case combinational transient and propagation delays"""
    for variation in config.variations('data_slews', 'loads', 'transient_sim_end_time'):
        for path in cell.paths():
            yield (measure_delays_for_path_with_criterion, cell, config, settings, variation, path, max)


@register('data_slews', 'loads', 'transient_sim_end_time')
def combinational_average(cell, config, settings):
    """Measure combinational transient and propagation delays using a uniform average"""
    for variation in config.variations('data_slews', 'loads', 'transient_sim_end_time'):
        for path in cell.paths():
            yield (measure_delays_for_path_with_criterion, cell, config, settings, variation, path, average)


def _run_single_condition(cell, config, settings, variation, path, state_map,
                          data_slew, load, t_sim_end, vdd, vss):
    """Build circuit, run simulation, extract measurements for one state condition.
    Returns (measurement_dict, measurement_names_set, pin_map, analysis, stable_pins_map_str)."""
    # Build the test circuit
    circuit = utils.init_circuit('comb_delay', cell.netlist, config.models,
                                 settings.named_nodes, settings.units)

    # Initialize device under test and wire up pins
    pin_map = utils.PinStateMap(cell.inputs, cell.outputs, state_map)
    connections = []
    measurements_list = []
    for pin in cell.pins_in_netlist_order():
        match pin.role:
            case Port.Role.LOGIC:
                if pin.name in pin_map.target_inputs:
                    connections.append(f'v{pin.name}')
                    (v_0, v_1) = (vss, vdd) if pin_map.target_inputs[pin.name] == '01' else (vdd, vss)
                    circuit.PieceWiseLinearVoltageSource(
                        pin.name, f'v{pin.name}', circuit.gnd,
                        values=utils.slew_pwl(v_0, v_1, data_slew, 3*data_slew,
                                              settings.logic_thresholds.low,
                                              settings.logic_thresholds.high))
                elif pin.name in pin_map.target_outputs:
                    connections.append(f'v{pin.name}')
                    circuit.C(pin.name, f'v{pin.name}', circuit.gnd, load)
                    for in_pin in pin_map.target_inputs:
                        if pin_map.target_inputs[in_pin] == '01':
                            in_direction = 'rise'
                            threshold_prop_0 = settings.logic_thresholds.rising
                        else:
                            in_direction = 'fall'
                            threshold_prop_0 = settings.logic_thresholds.falling
                        if pin_map.target_outputs[pin.name] == '01':
                            out_direction = 'rise'
                            threshold_prop_1 = settings.logic_thresholds.rising
                            threshold_tran_0 = settings.logic_thresholds.low
                            threshold_tran_1 = settings.logic_thresholds.high
                        else:
                            out_direction = 'fall'
                            threshold_prop_1 = settings.logic_thresholds.falling
                            threshold_tran_0 = settings.logic_thresholds.high
                            threshold_tran_1 = settings.logic_thresholds.low
                        prop_name = f'cell_{out_direction}__{in_pin}_to_{pin.name}'.lower()
                        measurements_list.append((
                            'tran', prop_name,
                            f'trig v(v{in_pin}) val={float(vdd*threshold_prop_0)} {in_direction}=1',
                            f'targ v(v{pin.name}) val={float(vdd*threshold_prop_1)} {out_direction}=1'))
                        tran_name = f'{out_direction}_transition__{in_pin}_to_{pin.name}'.lower()
                        measurements_list.append((
                            'tran', tran_name,
                            f'trig v(v{pin.name}) val={float(vdd*threshold_tran_0)} {out_direction}=1',
                            f'targ v(v{pin.name}) val={float(vdd*threshold_tran_1)} {out_direction}=1'))
                elif pin.name in pin_map.stable_inputs:
                    if pin_map.stable_inputs[pin.name] == '0':
                        connections.append(settings.primary_ground.name)
                    else:
                        connections.append(settings.primary_power.name)
                elif pin.name in pin_map.ignored_outputs:
                    connections.append('wfloat0')
                else:
                    raise ValueError(f'Unable to connect unrecognized logic pin {pin.name}')
            case Port.Role.POWER:
                connections.append(settings.primary_power.name)
            case Port.Role.GROUND:
                connections.append(settings.primary_ground.name)
            case Port.Role.NWELL:
                connections.append(settings.nwell.name)
            case Port.Role.PWELL:
                connections.append(settings.pwell.name)
            case _:
                raise ValueError(f'Unable to connect unrecognized pin {pin.name}')
    circuit.X('dut', cell.name, *connections)

    # Run simulation
    simulator = PySpice.Simulator.factory(simulator=settings.simulation.backend)
    simulation = simulator.simulation(
        circuit, temperature=settings.temperature, nominal_temperature=settings.temperature)
    simulation.options('autostop', 'nopage', 'nomod', post=1, ingold=2, trtol=1)
    for measure in measurements_list:
        simulation.measure(*measure, run=False)
    simulation.transient(step_time=data_slew/8, end_time=t_sim_end, run=False)

    stable_pins_map_str = ', '.join(['='.join([pin, state]) for pin, state in pin_map.stable_inputs.items()])
    try:
        analysis = simulator.run(simulation)
    except Exception as e:
        msg = f'Procedure failed for cell {cell.name} with variation {variation}, pin states {state_map}'
        if settings.debug:
            debug_path = settings.debug_dir / cell.name / __name__.split('.')[-1]
            debug_path.mkdir(parents=True, exist_ok=True)
            with open(debug_path / f'slew = {data_slew} load = {load}.sp', 'w', encoding='utf-8') as file:
                file.write(str(simulation))
        raise ProcedureFailedException(msg) from e

    # Extract measurements
    measurements = {}
    measurement_names = set()
    for measure_spec in measurements_list:
        name = measure_spec[1]
        if name in analysis.measurements:
            measurement_names.add(name)
            measurements[name] = float(analysis.measurements[name])

    return measurements, measurement_names, pin_map, analysis, stable_pins_map_str


def measure_delays_for_path_with_criterion(cell, config, settings, variation, path, criterion=max):
    """Given a particular path through the cell, find delays according to a selection criterion.

    This method tests all nonmasking conditions for the path through the cell from target_input to
    target_output with the given slew rate and capacitive load, then assigns the delay selected
    using the passed criterion function. Returns a liberty cell group with the delay information.

    The default criterion selects the worst-case (i.e. maximum) delay. This is in theory an overly
    pessimistic method of delay estimation. A more accurate method would be to perform a weighted
    average of each delay based on the likelihood of the corresponding state transition. However,
    at the time of writing this function, CharLib has no mechanism for accepting prior transition
    likelihood information.

    :param cell: A Cell object to test.
    :param config: A CellTestConfig object containing cell-specific test configuration details.
    :param settings: A CharacterizationSettings object containing library-wide configuration
                     details.
    :param variation: A dict containing test parameters for this configuration variation, such
                      as slew rates and loads.
    :param path: A list in the format [input_pin, input_transition, output_pin,
                 output_transtition] describing the path under test in the cell.
    :param criterion: A function which returns a single value given a list of numeric values.
                      Default max.
    """
    # Set up key parameters
    [input_pin, _, output_pin, _] = path
    data_slew = variation['data_slews'] * settings.units.time
    load = variation['loads'] * settings.units.capacitance
    t_sim_end = max(variation['transient_sim_end_time'] * settings.units.time, 1000*data_slew)
    vdd = settings.primary_power.voltage * settings.units.voltage
    vss = settings.primary_ground.voltage * settings.units.voltage

    # Measure delays for all nonmasking conditions
    analyses = {}
    measurement_names = set()
    for state_map in cell.nonmasking_conditions_for_path(*path):
        measurements, names, pin_map, analysis, stable_pins_map_str = _run_single_condition(
            cell, config, settings, variation, path, state_map,
            data_slew, load, t_sim_end, vdd, vss)
        analyses[stable_pins_map_str] = analysis
        measurement_names.update(names)

    # Select the worst-case delays and add to LUTs
    result = cell.liberty
    result.group('pin', output_pin).add_group('timing', f'/* {input_pin} */') # FIXME: This is a \
    # hack to allow multiple timing groups while the liberty API doesn't yet support multiple
    # groups with the same name and no id. In practice timing groups are distinguished by
    # their related_pin attribute.
    result.group('pin', output_pin).group('timing', f'/* {input_pin} */').add_attribute('related_pin', input_pin) # FIXME
    for name in measurement_names:
        # Get the worst delay & plot io
        if 'io' in config.plots:
            fig = plots.plot_io_voltages(analyses.values(), list(pin_map.target_inputs.keys()),
                                         list(pin_map.target_outputs.keys()),
                                         legend_labels=analyses.keys(),
                                         indicate_voltages=[settings.primary_power.voltage*settings.logic_thresholds.low,
                                                            settings.primary_power.voltage*settings.logic_thresholds.high])
            # FIXME: let user decide whether to show or save
            fig_path = settings.plots_dir / cell.name / 'io'
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f'{name} with slew = {data_slew} load = {load}.png') # FIXME: filetype should be configurable
            plt.close(fig)

        # Build LUT
        delay_measurements =[analysis.measurements[name] for analysis in analyses.values() if name in analysis.measurements]
        delay = criterion(delay_measurements) @ PySpice.Unit.u_s
        lut_name, meas_path = name.split('__')
        lut_template_size = f'{len(config.parameters["loads"])}x{len(config.parameters["data_slews"])}'
        lut = LookupTable(lut_name, f'delay_template_{lut_template_size}',
                          total_output_net_capacitance=[load.convert(settings.units.capacitance.prefixed_unit).value],
                          input_net_transition=[data_slew.convert(settings.units.time.prefixed_unit).value])
        lut.values[0,0] = delay.convert(settings.units.time.prefixed_unit).value
        result.group('pin', output_pin).group('timing', f'/* {input_pin} */').add_group(lut) # FIXME

    return result
