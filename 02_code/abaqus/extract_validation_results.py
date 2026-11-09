from __future__ import print_function

"""Extract full-circumference validation data from the two formal Abaqus ODBs.

Run with ``abaqus python extract_validation_results.py``.  The script is kept
compatible with the Python interpreter shipped with Abaqus 2022.
"""

from odbAccess import openOdb
import csv
import io
import json
import math
import os
import re


ROOT = r"G:\kongdong\lunwen4\research_v2_20261030"
ODB_DIR = os.environ.get("VALIDATION_ODB_DIR", os.path.join(ROOT, "03_fem", "results"))
OUT_DIR = os.environ.get("VALIDATION_OUT_DIR", os.path.join(ROOT, "04_validation", "fem_extracted"))
MODELS = tuple(os.environ.get(
    "VALIDATION_MODELS",
    "CAES-V2-ORTHO-LAYERED-ELASTIC-ROCK-B30-P45,"
    "CAES-V2-ORTHO-LAYERED-EP40-PSI10-B30-P45",
).split(","))
PRESSURE_STEPS = tuple(os.environ.get(
    "VALIDATION_PRESSURE_STEPS", "P010,P020,P030,P040,P045"
).split(","))
def pressure_mpa(step_name):
    match = re.match(r"P(\d+)", step_name)
    if not match:
        raise ValueError("Cannot infer pressure from step name %s" % step_name)
    return float(match.group(1)) / 10.0


PRESSURE = dict((name, pressure_mpa(name)) for name in PRESSURE_STEPS)
VOID_HALF_ANGLE_DEG = float(os.environ.get("VALIDATION_VOID_HALF_ANGLE_DEG", "15.0"))


def angle_deg(x, y):
    value = math.degrees(math.atan2(y, x))
    return value + 360.0 if value < 0.0 else value


def angular_distance(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def percentile(values, fraction):
    data = sorted(values)
    if not data:
        return None
    position = fraction * (len(data) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return data[low]
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def polygon_area(points):
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def instance_by_token(assembly, token):
    names = [name for name in assembly.instances.keys() if token in name.upper()]
    if len(names) != 1:
        raise RuntimeError("Could not uniquely find instance containing %s: %s" % (token, names))
    return assembly.instances[names[0]]


def geometry(instance):
    nodes = dict((node.label, (float(node.coordinates[0]), float(node.coordinates[1]))) for node in instance.nodes)
    elements = {}
    for element in instance.elements:
        points = [nodes[label] for label in element.connectivity]
        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        elements[element.label] = {
            "connectivity": tuple(element.connectivity),
            "x": x,
            "y": y,
            "r": math.hypot(x, y),
            "theta_deg": angle_deg(x, y),
            "area": polygon_area(points),
            "node_radii": tuple(math.hypot(*point) for point in points),
        }
    return nodes, elements


def nodal_vector(frame, instance, name):
    values = frame.fieldOutputs[name].getSubset(region=instance).values
    result = {}
    counts = {}
    for value in values:
        label = value.nodeLabel
        vector = tuple(float(item) for item in value.data)
        if label not in result:
            result[label] = [0.0] * len(vector)
            counts[label] = 0
        for i, item in enumerate(vector):
            result[label][i] += item
        counts[label] += 1
    for label in result:
        result[label] = tuple(item / counts[label] for item in result[label])
    return result


def summed_nforc(frame, instance):
    """Sum NFORC1/NFORC2 element-nodal components at each node."""
    result = {}
    for component, name in enumerate(("NFORC1", "NFORC2")):
        values = frame.fieldOutputs[name].getSubset(region=instance).values
        for value in values:
            label = value.nodeLabel
            if label not in result:
                result[label] = [0.0, 0.0]
            result[label][component] += float(value.data)
    return dict((label, tuple(items)) for label, items in result.items())


def circumferential_tributary_lengths(node_records):
    """Return periodic tributary arc length for (label, theta, radius) data."""
    ordered = sorted(node_records, key=lambda item: item[1])
    result = {}
    count = len(ordered)
    for i, (label, theta, radius) in enumerate(ordered):
        previous_theta = ordered[(i - 1) % count][1]
        next_theta = ordered[(i + 1) % count][1]
        backward = (theta - previous_theta) % (2.0 * math.pi)
        forward = (next_theta - theta) % (2.0 * math.pi)
        result[label] = radius * 0.5 * (backward + forward)
    return result


def element_scalar(frame, instance, name):
    if name not in frame.fieldOutputs:
        return {}, {}
    values = frame.fieldOutputs[name].getSubset(region=instance).values
    all_values = {}
    for value in values:
        all_values.setdefault(value.elementLabel, []).append(float(value.data))
    mean_values = dict((label, sum(items) / len(items)) for label, items in all_values.items())
    max_values = dict((label, max(items)) for label, items in all_values.items())
    return mean_values, max_values


def element_tensor_mean(frame, instance, name):
    if name not in frame.fieldOutputs:
        return {}
    values = frame.fieldOutputs[name].getSubset(region=instance).values
    grouped = {}
    for value in values:
        grouped.setdefault(value.elementLabel, []).append(tuple(float(item) for item in value.data))
    result = {}
    for label, items in grouped.items():
        width = len(items[0])
        result[label] = tuple(sum(item[i] for item in items) / len(items) for i in range(width))
    return result


def element_stress(frame, instance):
    values = frame.fieldOutputs["S"].getSubset(region=instance).values
    result = {}
    for value in values:
        data = tuple(float(item) for item in value.data)
        record = result.setdefault(value.elementLabel, {"data": [], "mises": [], "maxp": [], "minp": []})
        record["data"].append(data)
        try:
            record["mises"].append(float(value.mises))
            record["maxp"].append(float(value.maxPrincipal))
            record["minp"].append(float(value.minPrincipal))
        except Exception:
            pass
    averaged = {}
    for label, record in result.items():
        width = len(record["data"][0])
        averaged[label] = {
            "data": tuple(sum(item[i] for item in record["data"]) / len(record["data"]) for i in range(width)),
            "mises": sum(record["mises"]) / len(record["mises"]) if record["mises"] else None,
            "maxp": sum(record["maxp"]) / len(record["maxp"]) if record["maxp"] else None,
            "minp": sum(record["minp"]) / len(record["minp"]) if record["minp"] else None,
            "mises_ip_max": max(record["mises"]) if record["mises"] else None,
            "maxp_ip_max": max(record["maxp"]) if record["maxp"] else None,
        }
    return averaged


def polar_stress(data, theta):
    # Isotropic materials are output in global coordinates.  This conversion
    # is used for the steel seal and rock.  Concrete assessment uses stress
    # invariants because its components are output in local material axes.
    s11, s22, s12 = data[0], data[1], data[3]
    c, s = math.cos(theta), math.sin(theta)
    srr = c * c * s11 + s * s * s22 + 2.0 * c * s * s12
    stt = s * s * s11 + c * c * s22 - 2.0 * c * s * s12
    srt = (s22 - s11) * s * c + s12 * (c * c - s * s)
    return srr, stt, srt


def write_csv(path, header, rows):
    with open(path, "wb") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def extract_model(model):
    path = os.path.join(ODB_DIR, model + ".odb")
    odb = openOdb(path=path, readOnly=True)
    base_path = os.environ.get("VALIDATION_BASE_ODB", "")
    base_odb = openOdb(path=base_path, readOnly=True) if base_path else odb
    try:
        assembly = odb.rootAssembly
        lining = instance_by_token(assembly, "LINING_INSTANCE")
        sealing = instance_by_token(assembly, "SEALING_INSTANCE")
        rock = instance_by_token(assembly, "ROCK_INSTANCE")
        base_assembly = base_odb.rootAssembly
        base_lining = instance_by_token(base_assembly, "LINING_INSTANCE")
        base_sealing = instance_by_token(base_assembly, "SEALING_INSTANCE")
        base_rock = instance_by_token(base_assembly, "ROCK_INSTANCE")
        lining_nodes, lining_elements = geometry(lining)
        sealing_nodes, sealing_elements = geometry(sealing)
        rock_nodes, rock_elements = geometry(rock)

        # ADD-LINING deliberately requested a reduced output set and therefore
        # has no PEEQ field.  Lining activation carries no load, so the final
        # EXCAVATE frame is the correct operation-increment plasticity baseline.
        base_frame = base_odb.steps["EXCAVATE"].frames[-1]
        base_peeq_mean, base_peeq_max = element_scalar(base_frame, base_rock, "PEEQ")
        base_pe = element_tensor_mean(base_frame, base_rock, "PE")
        installation_frame = base_odb.steps["ADD-LINING"].frames[-1]
        installation_seal_u = nodal_vector(installation_frame, base_sealing, "U")
        installation_lining_u = nodal_vector(installation_frame, base_lining, "U")
        summaries = []

        for step_name in PRESSURE_STEPS:
            frame = odb.steps[step_name].frames[-1]
            seal_u = nodal_vector(frame, sealing, "U")
            lining_u = nodal_vector(frame, lining, "U")
            lining_nforc = summed_nforc(frame, lining)
            seal_s = element_stress(frame, sealing)
            lining_s = element_stress(frame, lining)
            rock_s = element_stress(frame, rock)
            peeq_mean, peeq_max = element_scalar(frame, rock, "PEEQ")
            final_pe = element_tensor_mean(frame, rock, "PE")

            seal_node_rows = []
            seal_inner_ur = []
            for label, (x, y) in sealing_nodes.items():
                radius = math.hypot(x, y)
                if abs(radius - 5.992) > 2.0e-4:
                    continue
                theta = math.atan2(y, x)
                u1 = seal_u[label][0] - installation_seal_u[label][0]
                u2 = seal_u[label][1] - installation_seal_u[label][1]
                ur = u1 * math.cos(theta) + u2 * math.sin(theta)
                ut = -u1 * math.sin(theta) + u2 * math.cos(theta)
                seal_node_rows.append((label, angle_deg(x, y), radius, ur, ut, u1, u2))
                seal_inner_ur.append(ur)
            seal_node_rows.sort(key=lambda row: row[1])

            lining_node_rows = []
            lining_outer_ur = []
            outer_node_geometry = []
            for label, (x, y) in lining_nodes.items():
                radius = math.hypot(x, y)
                if abs(radius - 6.5) > 2.0e-4:
                    continue
                theta = math.atan2(y, x)
                if theta < 0.0:
                    theta += 2.0 * math.pi
                outer_node_geometry.append((label, theta, radius))
            tributary = circumferential_tributary_lengths(outer_node_geometry)
            lining_outer_q = []
            lining_outer_void_q = []
            for label, theta, radius in outer_node_geometry:
                x, y = lining_nodes[label]
                u1 = lining_u[label][0] - installation_lining_u[label][0]
                u2 = lining_u[label][1] - installation_lining_u[label][1]
                ur = u1 * math.cos(theta) + u2 * math.sin(theta)
                ut = -u1 * math.sin(theta) + u2 * math.cos(theta)
                f1, f2 = lining_nforc[label][:2]
                fr = f1 * math.cos(theta) + f2 * math.sin(theta)
                ft = -f1 * math.sin(theta) + f2 * math.cos(theta)
                # For the lining element internal-force convention, positive
                # radial NFORC corresponds to an inward external tie traction.
                q = fr / tributary[label]
                tau = ft / tributary[label]
                is_void = angular_distance(angle_deg(x, y), 90.0) <= VOID_HALF_ANGLE_DEG + 1.0e-6
                lining_node_rows.append((label, angle_deg(x, y), radius, ur, ut, u1, u2,
                                         fr, ft, tributary[label], q, tau, int(is_void)))
                lining_outer_ur.append(ur)
                (lining_outer_void_q if is_void else lining_outer_q).append(q)
            lining_node_rows.sort(key=lambda row: row[1])

            seal_element_rows = []
            seal_mises = []
            for label, stress in seal_s.items():
                geo = sealing_elements[label]
                theta = math.atan2(geo["y"], geo["x"])
                srr, stt, srt = polar_stress(stress["data"], theta)
                seal_element_rows.append((label, geo["theta_deg"], geo["r"], srr, stt, srt,
                                          stress["mises"], stress["mises_ip_max"], stress["maxp"], stress["minp"]))
                seal_mises.append(stress["mises_ip_max"])
            seal_element_rows.sort(key=lambda row: (row[1], row[2]))

            lining_element_rows = []
            lining_maxp = []
            for label, stress in lining_s.items():
                geo = lining_elements[label]
                lining_element_rows.append((label, geo["theta_deg"], geo["r"], stress["maxp"],
                                            stress["maxp_ip_max"], stress["minp"], stress["mises"]))
                lining_maxp.append(stress["maxp_ip_max"])
            lining_element_rows.sort(key=lambda row: (row[1], row[2]))

            interface_rows = []
            contact_q = []
            void_q = []
            for label, stress in rock_s.items():
                geo = rock_elements[label]
                radii = geo["node_radii"]
                on_wall = sum(1 for radius in radii if abs(radius - 6.5) < 2.0e-4) >= 2
                if not on_wall or geo["r"] <= 6.5:
                    continue
                theta = math.atan2(geo["y"], geo["x"])
                srr, stt, srt = polar_stress(stress["data"], theta)
                q = -srr
                is_void = angular_distance(geo["theta_deg"], 90.0) <= VOID_HALF_ANGLE_DEG + 1.0e-6
                interface_rows.append((label, geo["theta_deg"], geo["r"], q, -srt, stt, int(is_void)))
                (void_q if is_void else contact_q).append(q)
            interface_rows.sort(key=lambda row: row[1])

            plastic_rows = []
            area_mean_1e6 = 0.0
            area_any_1e6 = 0.0
            max_delta = 0.0
            proxy_area_1e6 = 0.0
            proxy_max = 0.0
            if peeq_mean:
                for label, final_mean in peeq_mean.items():
                    if label not in rock_elements:
                        continue
                    delta_mean = max(0.0, final_mean - base_peeq_mean.get(label, 0.0))
                    delta_max = max(0.0, peeq_max.get(label, 0.0) - base_peeq_max.get(label, 0.0))
                    geo = rock_elements[label]
                    base_tensor = base_pe.get(label, (0.0, 0.0, 0.0, 0.0))
                    final_tensor = final_pe.get(label, base_tensor)
                    delta_tensor = tuple(final_tensor[i] - base_tensor[i] for i in range(len(final_tensor)))
                    theta = math.atan2(geo["y"], geo["x"])
                    drr, dtt, drt = polar_stress(delta_tensor, theta)
                    dzz = delta_tensor[2]
                    plastic_proxy = math.sqrt(drr * drr + dtt * dtt + dzz * dzz + 2.0 * drt * drt)
                    if delta_mean > 1.0e-6:
                        area_mean_1e6 += geo["area"]
                    if delta_max > 1.0e-6:
                        area_any_1e6 += geo["area"]
                    max_delta = max(max_delta, delta_max)
                    if plastic_proxy > 1.0e-6:
                        proxy_area_1e6 += geo["area"]
                    proxy_max = max(proxy_max, plastic_proxy)
                    if delta_max > 0.0 or plastic_proxy > 0.0:
                        plastic_rows.append((label, geo["theta_deg"], geo["r"], geo["area"],
                                             base_peeq_mean.get(label, 0.0), final_mean, delta_mean, delta_max,
                                             plastic_proxy))
                plastic_rows.sort(key=lambda row: (row[2], row[1]))

            stem = "%s_%s" % (model, step_name)
            write_csv(os.path.join(OUT_DIR, stem + "_seal_inner_u.csv"),
                      ("node", "theta_deg", "r_m", "ur_m", "ut_m", "u1_m", "u2_m"), seal_node_rows)
            write_csv(os.path.join(OUT_DIR, stem + "_lining_outer_u.csv"),
                      ("node", "theta_deg", "r_m", "ur_m", "ut_m", "u1_m", "u2_m", "nforc_r_n_per_m",
                       "nforc_t_n_per_m", "tributary_length_m", "q_from_nforc_pa", "tau_from_nforc_pa",
                       "void_sector"), lining_node_rows)
            write_csv(os.path.join(OUT_DIR, stem + "_seal_stress.csv"),
                      ("element", "theta_deg", "r_m", "srr_pa", "stt_pa", "srt_pa", "mises_mean_pa",
                       "mises_ip_max_pa", "max_principal_mean_pa", "min_principal_mean_pa"), seal_element_rows)
            write_csv(os.path.join(OUT_DIR, stem + "_lining_stress.csv"),
                      ("element", "theta_deg", "r_m", "max_principal_mean_pa", "max_principal_ip_max_pa",
                       "min_principal_mean_pa", "mises_mean_pa"), lining_element_rows)
            write_csv(os.path.join(OUT_DIR, stem + "_rock_wall_stress.csv"),
                      ("element", "theta_deg", "r_m", "q_pa", "tau_pa", "stt_pa", "void_sector"), interface_rows)
            if plastic_rows:
                write_csv(os.path.join(OUT_DIR, stem + "_rock_delta_peeq.csv"),
                          ("element", "theta_deg", "r_m", "area_m2", "peeq_base_mean", "peeq_final_mean",
                           "delta_peeq_mean", "delta_peeq_ip_max", "delta_plastic_tensor_proxy"), plastic_rows)

            summaries.append({
                "step": step_name,
                "pressure_mpa": PRESSURE[step_name],
                "seal_inner_ur_max_m": max(seal_inner_ur),
                "seal_inner_ur_min_m": min(seal_inner_ur),
                "lining_outer_ur_max_m": max(lining_outer_ur),
                "lining_outer_ur_min_m": min(lining_outer_ur),
                "seal_mises_max_pa": max(seal_mises),
                "seal_mises_p99_pa": percentile(seal_mises, 0.99),
                "lining_max_principal_max_pa": max(lining_maxp),
                "lining_max_principal_p99_pa": percentile(lining_maxp, 0.99),
                "rock_wall_contact_q_mean_pa": sum(contact_q) / len(contact_q),
                "rock_wall_contact_q_max_pa": max(contact_q),
                "rock_wall_void_abs_q_mean_pa": sum(abs(value) for value in void_q) / len(void_q),
                "lining_nforc_contact_q_mean_pa": sum(lining_outer_q) / len(lining_outer_q),
                "lining_nforc_contact_q_max_pa": max(lining_outer_q),
                "lining_nforc_contact_q_min_pa": min(lining_outer_q),
                "lining_nforc_void_abs_q_mean_pa": sum(abs(value) for value in lining_outer_void_q) / len(lining_outer_void_q),
                "rock_delta_peeq_area_mean_gt_1e-6_m2": area_mean_1e6,
                "rock_delta_peeq_area_any_ip_gt_1e-6_m2": area_any_1e6,
                "rock_delta_peeq_max": max_delta,
                "rock_delta_plastic_proxy_area_gt_1e-6_m2": proxy_area_1e6,
                "rock_delta_plastic_proxy_max": proxy_max,
            })

        return {
            "model": model,
            "steps": list(odb.steps.keys()),
            "instances": list(assembly.instances.keys()),
            "summary": summaries,
        }
    finally:
        if base_odb is not odb:
            base_odb.close()
        odb.close()


def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    all_results = [extract_model(model) for model in MODELS]
    path = os.path.join(OUT_DIR, "fem_validation_summary.json")
    # Abaqus 2022 ships Python 2.7; ASCII JSON avoids the str/unicode mismatch
    # of json.dump with an io.open text wrapper in that interpreter.
    with open(path, "w") as handle:
        json.dump(all_results, handle, ensure_ascii=True, indent=2)
    print(path)


if __name__ == "__main__":
    main()
