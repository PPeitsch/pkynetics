from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pkynetics.data_preprocessing.common_preprocessing import smooth_data
from pkynetics.technique_analysis.utilities import detect_segment_direction

ReturnDict = Dict[str, Union[float, NDArray[np.float64], Dict[str, float]]]


def extrapolate_linear_segments(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], np.poly1d, np.poly1d]:
    """
    Extrapolate linear segments before and after the transformation.

    Args:
        temperature: Array of temperature values
        strain: Array of strain values
        start_temp: Start temperature of the transformation
        end_temp: End temperature of the transformation

    Returns:
        Tuple containing:
        - Extrapolated strain values before transformation
        - Extrapolated strain values after transformation
        - Polynomial function for before extrapolation
        - Polynomial function for after extrapolation

    Raises:
        ValueError: If temperatures are invalid or if insufficient data for fitting
    """
    if start_temp >= end_temp:
        raise ValueError("Start temperature must be less than end temperature")
    if not (temperature.min() <= start_temp <= temperature.max()):
        raise ValueError("Start temperature outside data range")
    if not (temperature.min() <= end_temp <= temperature.max()):
        raise ValueError("End temperature outside data range")

    before_mask = temperature < start_temp
    after_mask = temperature > end_temp

    min_points = 5
    if np.sum(before_mask) < min_points or np.sum(after_mask) < min_points:
        raise ValueError(
            f"Insufficient points for fitting. Need at least {min_points} points in each region."
        )

    try:
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)

        before_extrapolation = np.poly1d(before_fit)
        after_extrapolation = np.poly1d(after_fit)

    except np.linalg.LinAlgError:
        raise ValueError("Unable to perform linear fit on the data segments")

    before_values = before_extrapolation(temperature)
    after_values = after_extrapolation(temperature)

    return before_values, after_values, before_extrapolation, after_extrapolation


def find_optimal_margin(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    min_r2: float = 0.99,
    min_points: int = 10,
) -> float:
    """
    Determine the optimal margin percentage for linear segment fitting.

    Args:
        temperature: Temperature data array
        strain: Strain data array
        min_r2: Minimum R² value for acceptable linear fit (default: 0.99)
        min_points: Minimum number of points required for fitting (default: 10)

    Returns:
        float: Optimal margin percentage (between 0.1 and 0.4)

    Raises:
        ValueError: If no acceptable margin is found or if data is insufficient
    """
    if len(temperature) < min_points * 2:
        raise ValueError(
            f"Insufficient data points. Need at least {min_points * 2} points."
        )

    margins = np.linspace(0.1, 0.4, 7)  # Test margins from 10% to 40%
    best_margin: Optional[float] = None
    best_r2: float = 0.0

    for margin in margins:
        n_points = int(len(temperature) * margin)
        if n_points < min_points:
            continue

        start_mask = temperature <= (
            temperature.min() + (temperature.max() - temperature.min()) * margin
        )
        if np.sum(start_mask) < min_points:
            continue

        end_mask = temperature >= (
            temperature.max() - (temperature.max() - temperature.min()) * margin
        )
        if np.sum(end_mask) < min_points:
            continue

        try:
            p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
            r2_start = calculate_r2(
                temperature[start_mask], strain[start_mask], p_start
            )

            p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
            r2_end = calculate_r2(temperature[end_mask], strain[end_mask], p_end)

            avg_r2 = (r2_start + r2_end) / 2

            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_margin = margin

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_margin is None or best_r2 < min_r2:
        if best_margin is not None:
            return float(best_margin)
        return 0.2

    return float(best_margin)


# Corregir cómo se calcula la fracción transformada para enfriamiento
def calculate_transformed_fraction_lever(
        temperature: NDArray[np.float64],
        strain: NDArray[np.float64],
        start_temp: float,
        end_temp: float,
        margin_percent: float = 0.2,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate transformed fraction using the lever rule method."""
    # Detectar dirección
    is_cooling = detect_segment_direction(temperature, strain)

    # Validar rango
    temp_min, temp_max = min(temperature), max(temperature)
    if not (temp_min <= start_temp <= temp_max and temp_min <= end_temp <= temp_max):
        raise ValueError("Temperature values outside data range")

    # Determinar regiones para ajuste lineal
    temp_range = temp_max - temp_min
    fit_range = temp_range * margin_percent

    # Ajustar máscaras según dirección
    if is_cooling:
        before_mask = temperature >= (temp_max - fit_range)
        after_mask = temperature <= (temp_min + fit_range)
    else:
        before_mask = temperature <= (temp_min + fit_range)
        after_mask = temperature >= (temp_max - fit_range)

    # Validar puntos suficientes
    min_points = 5
    if np.sum(before_mask) < min_points or np.sum(after_mask) < min_points:
        raise ValueError(f"Insufficient points for fitting in linear regions")

    # Ajuste lineal
    try:
        before_fit = np.polyfit(temperature[before_mask], strain[before_mask], 1)
        after_fit = np.polyfit(temperature[after_mask], strain[after_mask], 1)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to perform linear fit on the data segments")

    # Extrapolaciones
    before_extrap = np.polyval(before_fit, temperature)
    after_extrap = np.polyval(after_fit, temperature)

    # Inicializar fracción
    transformed_fraction = np.zeros_like(strain)

    # Calcular para región de transformación
    if is_cooling:
        mask = (temperature <= start_temp) & (temperature >= end_temp)
    else:
        mask = (temperature >= start_temp) & (temperature <= end_temp)

    # PUNTO CLAVE: Para enfriamiento, necesitamos calcular la altura de manera diferente
    if is_cooling:
        # Para enfriamiento: cambiar la relación para que sea 1.0 en start_temp y 0.0 en end_temp
        # ESTA ES LA CORRECCIÓN CRÍTICA:
        # Usamos (1 - ratio) para invertir correctamente la relación para enfriamiento
        height_total = after_extrap[mask] - before_extrap[mask]
        height_current = strain[mask] - before_extrap[mask]
        valid_total = height_total != 0
        # Invertir la relación dentro del cálculo mismo
        transformed_fraction[mask] = np.where(valid_total,
                                              1.0 - (height_current / height_total),
                                              1.0)
    else:
        # Para calentamiento: cálculo normal
        height_total = after_extrap[mask] - before_extrap[mask]
        height_current = strain[mask] - before_extrap[mask]
        valid_total = height_total != 0
        transformed_fraction[mask] = np.where(valid_total, height_current / height_total, 0)

    # Valores fuera del rango de transformación
    if is_cooling:
        transformed_fraction[temperature > start_temp] = 1.0
        transformed_fraction[temperature < end_temp] = 0.0
    else:
        transformed_fraction[temperature < start_temp] = 0.0
        transformed_fraction[temperature > end_temp] = 1.0

    return np.clip(transformed_fraction, 0, 1), before_extrap, after_extrap


def analyze_dilatometry_curve(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    method: str = "lever",
    margin_percent: float = 0.2,
) -> ReturnDict:
    """Analyze the dilatometry curve to extract key parameters."""
    if method == "lever":
        return lever_method(temperature, strain, margin_percent)
    elif method == "tangent":
        return tangent_method(temperature, strain, margin_percent)
    else:
        raise ValueError(f"Unsupported method: {method}")


def lever_method(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    margin_percent: float = 0.2,
) -> ReturnDict:
    """Analyze dilatometry curve using the lever rule method."""
    is_cooling = detect_segment_direction(temperature, strain)
    start_temp, end_temp = find_inflection_points(temperature, strain, is_cooling)

    transformed_fraction, before_extrap, after_extrap = (
        calculate_transformed_fraction_lever(
            temperature, strain, start_temp, end_temp, margin_percent
        )
    )

    # Agregar is_cooling al llamado
    mid_temp = find_midpoint_temperature(
        temperature, transformed_fraction, start_temp, end_temp, is_cooling
    )

    return {
        "start_temperature": float(start_temp),
        "end_temperature": float(end_temp),
        "mid_temperature": float(mid_temp),
        "transformed_fraction": transformed_fraction,
        "before_extrapolation": before_extrap,
        "after_extrapolation": after_extrap,
        "is_cooling": is_cooling,  # Agregar al resultado
    }


def tangent_method(
        temperature: NDArray[np.float64],
        strain: NDArray[np.float64],
        margin_percent: Optional[float] = None,
        deviation_threshold: Optional[float] = None,
) -> ReturnDict:
    """Analyze dilatometry curve using the tangent method."""
    temperature = np.asarray(temperature)
    strain = np.asarray(strain)

    # Detectar dirección del segmento
    is_cooling = detect_segment_direction(temperature, strain)

    if margin_percent is None:
        margin_percent = find_optimal_margin(temperature, strain)

    # Pasar is_cooling a todas las funciones relevantes
    start_mask, end_mask = get_linear_segment_masks(temperature, float(margin_percent), is_cooling)
    p_start, p_end = fit_linear_segments(temperature, strain, start_mask, end_mask)
    pred_start, pred_end = get_extrapolated_values(temperature, p_start, p_end)

    final_deviation_threshold = float(
        deviation_threshold
        if deviation_threshold is not None
        else calculate_deviation_threshold(
            strain, pred_start, pred_end, start_mask, end_mask
        )
    )

    start_idx, end_idx = find_transformation_points(
        temperature, strain, pred_start, pred_end, final_deviation_threshold, is_cooling
    )

    transformed_fraction = calculate_transformed_fraction(
        strain, pred_start, pred_end, start_idx, end_idx, is_cooling
    )

    mid_temp = find_midpoint_temperature(
        temperature, transformed_fraction, temperature[start_idx], temperature[end_idx], is_cooling
    )

    fit_quality = calculate_fit_quality(
        temperature,
        strain,
        p_start,
        p_end,
        start_mask,
        end_mask,
        float(margin_percent),
        final_deviation_threshold,
    )

    return {
        "start_temperature": float(temperature[start_idx]),
        "end_temperature": float(temperature[end_idx]),
        "mid_temperature": float(mid_temp),
        "transformed_fraction": transformed_fraction,
        "before_extrapolation": pred_start,
        "after_extrapolation": pred_end,
        "fit_quality": fit_quality,
        "is_cooling": is_cooling,
    }


def find_inflection_points(
        temperature: NDArray[np.float64],
        strain: NDArray[np.float64],
        is_cooling: bool = False
) -> Tuple[float, float]:
    """Find transformation points where curve deviates from extrapolations."""
    # Suavizar datos
    smooth_strain = smooth_data(strain)

    # Obtener segmentos lineales iniciales (10-30% de los extremos)
    temp_range = max(temperature) - min(temperature)
    margin = temp_range * 0.3

    # Definir regiones lineales según si es enfriamiento o calentamiento
    if is_cooling:
        high_temp_mask = temperature >= (max(temperature) - margin)  # Región inicial (alta temp)
        low_temp_mask = temperature <= (min(temperature) + margin)  # Región final (baja temp)
    else:
        low_temp_mask = temperature <= (min(temperature) + margin)  # Región inicial (baja temp)
        high_temp_mask = temperature >= (max(temperature) - margin)  # Región final (alta temp)

    # Ajustar líneas tangentes en regiones lineales
    high_temp_fit = np.polyfit(temperature[high_temp_mask], smooth_strain[high_temp_mask], 1)
    low_temp_fit = np.polyfit(temperature[low_temp_mask], smooth_strain[low_temp_mask], 1)

    # Calcular extrapolaciones completas
    high_temp_line = np.polyval(high_temp_fit, temperature)
    low_temp_line = np.polyval(low_temp_fit, temperature)

    # Calcular residuos (diferencias entre curva real y extrapolaciones)
    high_temp_residuals = np.abs(smooth_strain - high_temp_line)
    low_temp_residuals = np.abs(smooth_strain - low_temp_line)

    # Calcular ruido base en regiones lineales
    noise_high = np.std(high_temp_residuals[high_temp_mask]) * 3.0
    noise_low = np.std(low_temp_residuals[low_temp_mask]) * 3.0

    # Ordenar índices por temperatura (de acuerdo a la dirección)
    if is_cooling:
        temp_sorted_indices = np.argsort(-temperature)  # descendente
    else:
        temp_sorted_indices = np.argsort(temperature)  # ascendente

    # Buscar punto de inicio - primer punto donde residuo excede significativamente el ruido base
    # y continúa creciendo
    start_idx = None
    for i in range(len(temp_sorted_indices) // 10, len(temp_sorted_indices) * 9 // 10):
        idx = temp_sorted_indices[i]

        # Verificar si residuo excede umbral
        if high_temp_residuals[idx] > noise_high:
            # Confirmar con puntos adyacentes (tendencia creciente)
            if i + 2 < len(temp_sorted_indices):
                next_idx1 = temp_sorted_indices[i + 1]
                next_idx2 = temp_sorted_indices[i + 2]

                if (high_temp_residuals[next_idx1] > high_temp_residuals[idx] and
                        high_temp_residuals[next_idx2] > high_temp_residuals[next_idx1]):

                    # Buscar hacia atrás para encontrar punto exacto de inicio
                    for j in range(i - 1, 0, -1):
                        back_idx = temp_sorted_indices[j]
                        if high_temp_residuals[back_idx] < noise_high:
                            start_idx = temp_sorted_indices[j + 1]
                            break
                    else:
                        start_idx = idx
                    break

    # Buscar punto final de manera similar
    end_idx = None
    for i in range(len(temp_sorted_indices) * 9 // 10, len(temp_sorted_indices) // 10, -1):
        idx = temp_sorted_indices[i]

        if low_temp_residuals[idx] > noise_low:
            if i - 2 >= 0:
                prev_idx1 = temp_sorted_indices[i - 1]
                prev_idx2 = temp_sorted_indices[i - 2]

                if (low_temp_residuals[prev_idx1] > low_temp_residuals[idx] and
                        low_temp_residuals[prev_idx2] > low_temp_residuals[prev_idx1]):

                    # Buscar hacia adelante para punto exacto
                    for j in range(i + 1, len(temp_sorted_indices) - 1):
                        fwd_idx = temp_sorted_indices[j]
                        if low_temp_residuals[fwd_idx] < noise_low:
                            end_idx = temp_sorted_indices[j - 1]
                            break
                    else:
                        end_idx = idx
                    break

    # Fallback si no se encuentran puntos claros
    if start_idx is None:
        start_idx = temp_sorted_indices[len(temp_sorted_indices) // 4]
    if end_idx is None:
        end_idx = temp_sorted_indices[3 * len(temp_sorted_indices) // 4]

    # Convertir a temperaturas
    start_temp = float(temperature[start_idx])
    end_temp = float(temperature[end_idx])

    # Asegurar orden correcto según dirección
    if is_cooling:
        if start_temp < end_temp:
            start_temp, end_temp = end_temp, start_temp
    else:
        if start_temp > end_temp:
            start_temp, end_temp = end_temp, start_temp

    return start_temp, end_temp


def find_midpoint_temperature(
    temperature: NDArray[np.float64],
    transformed_fraction: NDArray[np.float64],
    start_temp: float,
    end_temp: float,
    is_cooling: bool = False
) -> float:
    """Find temperature at 50% transformation."""
    # Asegurar el orden correcto para la máscara según dirección
    if is_cooling:
        # En enfriamiento, start_temp > end_temp
        temp_high, temp_low = max(start_temp, end_temp), min(start_temp, end_temp)
        mask = (temperature <= temp_high) & (temperature >= temp_low)
    else:
        # En calentamiento, start_temp < end_temp
        temp_low, temp_high = min(start_temp, end_temp), max(start_temp, end_temp)
        mask = (temperature >= temp_low) & (temperature <= temp_high)

    valid_fraction = transformed_fraction[mask]
    valid_temp = temperature[mask]

    if len(valid_fraction) == 0:
        # Fallback: interpolación lineal entre puntos de inicio y fin
        from scipy.interpolate import interp1d
        try:
            temps = np.array([start_temp, end_temp])
            fracs = np.array([0.0, 1.0])
            if is_cooling:
                fracs = np.array([1.0, 0.0])
            f = interp1d(fracs, temps)
            return float(f(0.5))
        except:
            # Si todo falla, simplemente promediar
            return float((start_temp + end_temp) / 2)

    # Encontrar el punto más cercano al 50% de transformación
    mid_idx = np.argmin(np.abs(valid_fraction - 0.5))
    return float(valid_temp[mid_idx])


def get_linear_segment_masks(
        temperature: NDArray[np.float64],
        margin_percent: float,
        is_cooling: bool = False
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Get masks for linear segments at start and end."""
    temp_range = max(temperature) - min(temperature)
    margin = temp_range * margin_percent

    if is_cooling:
        # Para enfriamiento: segmento inicial en temperaturas altas, final en bajas
        start_mask = temperature >= (max(temperature) - margin)
        end_mask = temperature <= (min(temperature) + margin)
    else:
        # Para calentamiento: segmento inicial en temperaturas bajas, final en altas
        start_mask = temperature <= (min(temperature) + margin)
        end_mask = temperature >= (max(temperature) - margin)

    return start_mask, end_mask


def fit_linear_segments(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit linear functions to start and end segments."""
    p_start = np.polyfit(temperature[start_mask], strain[start_mask], 1)
    p_end = np.polyfit(temperature[end_mask], strain[end_mask], 1)
    return p_start, p_end


def get_extrapolated_values(
    temperature: NDArray[np.float64],
    p_start: NDArray[np.float64],
    p_end: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate extrapolated values using linear fits."""
    pred_start = np.polyval(p_start, temperature)
    pred_end = np.polyval(p_end, temperature)
    return pred_start, pred_end


# Corregir cómo se detectan los puntos de transformación para el método tangente
def find_transformation_points(
        temperature: NDArray[np.float64],
        strain: NDArray[np.float64],
        pred_start: NDArray[np.float64],
        pred_end: NDArray[np.float64],
        deviation_threshold: float,
        is_cooling: bool = False
) -> Tuple[int, int]:
    """Find transformation start and end points."""
    # Calcular desviaciones entre curva real y extrapolaciones
    dev_start = np.abs(strain - pred_start)
    dev_end = np.abs(strain - pred_end)

    # Suavizar desviaciones para detección más robusta
    from scipy.signal import savgol_filter
    window = min(max(int(len(temperature) * 0.05), 5), 21)  # Ventana razonable
    if window % 2 == 0:
        window += 1  # Asegurar ventana impar

    smooth_dev_start = savgol_filter(dev_start, window, 2)
    smooth_dev_end = savgol_filter(dev_end, window, 2)

    # Calcular gradientes para encontrar puntos de desviación significativa
    grad_start = np.gradient(smooth_dev_start)
    grad_end = np.gradient(smooth_dev_end)

    # Ordenar temperatura según dirección
    if is_cooling:
        temp_indices = np.argsort(-temperature)  # Descendente
    else:
        temp_indices = np.argsort(temperature)  # Ascendente

    # Buscar puntos donde el gradiente supera un umbral
    grad_threshold = np.std(grad_start) * 0.75

    # Para enfriamiento, buscar desde temperatura alta
    if is_cooling:
        # Desde alto a bajo para start_idx
        for i in temp_indices:
            if grad_start[i] > grad_threshold and smooth_dev_start[i] > deviation_threshold:
                start_idx = i
                break
        else:
            start_idx = temp_indices[len(temp_indices) // 4]

        # Desde bajo a alto para end_idx
        for i in temp_indices[::-1]:
            if grad_end[i] > grad_threshold and smooth_dev_end[i] > deviation_threshold:
                end_idx = i
                break
        else:
            end_idx = temp_indices[3 * len(temp_indices) // 4]
    else:
        # Para calentamiento, procedimiento similar pero en dirección opuesta
        for i in temp_indices:
            if grad_start[i] > grad_threshold and smooth_dev_start[i] > deviation_threshold:
                start_idx = i
                break
        else:
            start_idx = temp_indices[len(temp_indices) // 4]

        for i in temp_indices[::-1]:
            if grad_end[i] > grad_threshold and smooth_dev_end[i] > deviation_threshold:
                end_idx = i
                break
        else:
            end_idx = temp_indices[3 * len(temp_indices) // 4]

    # Asegurar orden correcto
    if is_cooling:
        if temperature[start_idx] < temperature[end_idx]:
            start_idx, end_idx = end_idx, start_idx
    else:
        if temperature[start_idx] > temperature[end_idx]:
            start_idx, end_idx = end_idx, start_idx

    return start_idx, end_idx


def calculate_deviation_threshold(
    strain: NDArray[np.float64],
    pred_start: NDArray[np.float64],
    pred_end: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
) -> float:
    """Calculate threshold for deviation detection."""
    start_residuals = np.abs(strain[start_mask] - pred_start[start_mask])
    end_residuals = np.abs(strain[end_mask] - pred_end[end_mask])
    return float(3 * max(float(np.std(start_residuals)), float(np.std(end_residuals))))


def find_deviation_point(
    deviations: NDArray[np.bool_], window: int, forward: bool = True
) -> int:
    """Find point where deviation becomes significant."""
    if forward:
        cum_dev = np.convolve(deviations, np.ones(window) / window, mode="valid")
        return int(np.argmax(cum_dev > 0.8) + window // 2)
    else:
        cum_dev = np.convolve(deviations[::-1], np.ones(window) / window, mode="valid")
        return int(len(deviations) - np.argmax(cum_dev > 0.8) - window // 2)


def calculate_transformed_fraction(
        strain: NDArray[np.float64],
        pred_start: NDArray[np.float64],
        pred_end: NDArray[np.float64],
        start_idx: int,
        end_idx: int,
        is_cooling: bool = False,
) -> NDArray[np.float64]:
    """Calculate transformed fraction."""
    transformed_fraction = np.zeros_like(strain)
    transformation_region = slice(start_idx, end_idx + 1)

    height_total = pred_end[transformation_region] - pred_start[transformation_region]
    height_current = strain[transformation_region] - pred_start[transformation_region]
    transformed_fraction[transformation_region] = height_current / height_total

    if is_cooling:
        transformed_fraction = 1 - transformed_fraction
        transformed_fraction[end_idx + 1:] = 0.0
    else:
        transformed_fraction[end_idx + 1:] = 1.0

    return np.clip(transformed_fraction, 0, 1)


def calculate_fit_quality(
    temperature: NDArray[np.float64],
    strain: NDArray[np.float64],
    p_start: NDArray[np.float64],
    p_end: NDArray[np.float64],
    start_mask: NDArray[np.bool_],
    end_mask: NDArray[np.bool_],
    margin_percent: float,
    deviation_threshold: float,
) -> Dict[str, float]:
    """Calculate quality metrics for the analysis."""
    r2_start = float(calculate_r2(temperature[start_mask], strain[start_mask], p_start))
    r2_end = float(calculate_r2(temperature[end_mask], strain[end_mask], p_end))

    return {
        "r2_start": r2_start,
        "r2_end": r2_end,
        "margin_used": float(margin_percent),
        "deviation_threshold": float(deviation_threshold),
    }


def calculate_r2(
    x: NDArray[np.float64], y: NDArray[np.float64], p: NDArray[np.float64]
) -> float:
    """Calculate R² value for a linear fit."""
    y_pred = np.polyval(p, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1 - (ss_res / ss_tot))
