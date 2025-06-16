import numpy as np
from skyfield.api import load, wgs84
from scipy.spatial.transform import Rotation as R
import pyvista as pv
from bs4 import BeautifulSoup
import requests
import re
from datetime import datetime as dt_datetime
import itertools
import asyncio
from playwright.async_api import async_playwright


# --- ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ КООРДИНАТ ЧЕРЕЗ SELENIUM/PLAYWRIGHT ---
async def get_coordinates_with_playwright(url_target='https://r4uab.ru/satdb/cubesx-sirius-hse/'):
    """
    Извлекает широту, долготу и высоту со страницы с помощью Playwright,
    работая с iframe.
    """
    print("Playwright: Запуск для получения координат...")
    lat_val, lon_val, alt_val = None, None, None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)  # или headless=False для отладки
            page = await browser.new_page()
            await page.goto(url_target, timeout=30000)

            iframe_selector = 'iframe[src="https://r4uab.ru/content/modules/sattrack/?catalog_number=47951"]'
            print(f"Playwright: Ожидание iframe с селектором: {iframe_selector}")
            try:
                await page.wait_for_selector(iframe_selector, state='visible', timeout=20000)
                iframe_element = await page.query_selector(iframe_selector)
            except Exception as e_iframe_wait:
                print(f"Playwright: Ошибка ожидания iframe: {e_iframe_wait}")
                iframe_element = None

            if iframe_element:
                print("Playwright: Iframe найден.")
                frame = await iframe_element.content_frame()
                if frame:
                    print("Playwright: Content frame получен.")
                    await frame.wait_for_selector('#data_panel', state='visible', timeout=20000)
                    print("Playwright: #data_panel найдена в iframe.")

                    raw_latitude_text = await frame.inner_text('.latitude')
                    raw_longitude_text = await frame.inner_text('.longitude')
                    raw_altitude_text = await frame.inner_text('.altitude')

                    print(
                        f"Playwright: Сырые данные: Lat='{raw_latitude_text}', Lon='{raw_longitude_text}', Alt='{raw_altitude_text}'")

                    latitude_text = raw_latitude_text.replace("Lat:", "").strip()
                    longitude_text = raw_longitude_text.replace("Lng:", "").strip()
                    altitude_text = raw_altitude_text.replace("Alt:", "").strip()

                    match_lat = re.search(r"(\d+)°\s*([\d.]+)[′']\s*([NS])", latitude_text)  # Добавил ' для апострофа
                    if match_lat:
                        deg, minute, direction = match_lat.groups()
                        lat_val = float(deg) + float(minute) / 60.0
                        if direction == 'S': lat_val *= -1

                    match_lon = re.search(r"(\d+)°\s*([\d.]+)[′']\s*([EW])", longitude_text)  #' для апострофа
                    if match_lon:
                        deg, minute, direction = match_lon.groups()
                        lon_val = float(deg) + float(minute) / 60.0
                        if direction == 'W':
                            lon_val *= -1

                    match_alt = re.search(r"([\d.]+)\s*км", altitude_text)
                    if match_alt:
                        alt_val = float(match_alt.group(1))

                    if lat_val is not None and lon_val is not None and alt_val is not None:
                        print(
                            f"Playwright: Координаты извлечены: Lat={lat_val:.3f}, Lon={lon_val:.3f}, Alt={alt_val:.3f}")
                    else:
                        print("Playwright: Ошибка парсинга одной или нескольких координат после очистки.")
                        if match_lat is None: print(f"  Ошибка парсинга широты из очищенной строки: '{latitude_text}'")
                        if match_lon is None: print(
                            f"  Ошибка парсинга долготы из очищенной строки: '{longitude_text}'")
                        if match_alt is None: print(f"  Ошибка парсинга высоты из очищенной строки: '{altitude_text}'")
                else:
                    print("Playwright: Не удалось получить content_frame() из iframe.")
            else:
                print("Playwright: Iframe не найден по селектору.")
            await browser.close()
            print("Playwright: Браузер закрыт.")
    except Exception as e:
        print(f"Playwright: Общая ошибка - {e}")
    return lat_val, lon_val, alt_val

print("Запрос данных телеметрии с сайта r4uab.ru...")
url_telemetry = "https://r4uab.ru/satdb/cubesx-sirius-hse/"  # Отдельный URL для телеметрии
headers_bs4 = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
Time_value = None;
Date_TLM = None;
electricity1 = None;
electricity2 = None;
electricity3 = None
try:
    response_bs4 = requests.get(url_telemetry, headers=headers_bs4, timeout=15)
    response_bs4.raise_for_status()
    soup_bs4 = BeautifulSoup(response_bs4.text, 'html.parser')
    content_price_blocks = soup_bs4.find_all('div', class_='content_price_nov')
    for block in content_price_blocks:
        header_div = block.find('div', class_='name_price_nov')
        value_span_container = block.find('div', class_='price_opisanie')
        if header_div and value_span_container:
            value_span = value_span_container.find('span')
            if value_span:
                header_text = header_div.get_text(strip=True)
                value_text = value_span.get_text(strip=True)
                if 'Время ТЛМ' == header_text:
                    Time_value = value_text
                elif 'Дата ТЛМ' == header_text:
                    Date_TLM = value_text
                elif 'Ток СБ №1' == header_text:
                    electricity1 = value_text
                elif 'Ток СБ №2' == header_text:
                    electricity2 = value_text
                elif 'Ток СБ №3' == header_text:
                    electricity3 = value_text
    if Date_TLM and Time_value:
        print(f"Данные телеметрии с r4uab.ru получены: Дата ТЛМ={Date_TLM}, Время ТЛМ={Time_value}")
    else:
        print("Не все данные для даты/времени ТЛМ были найдены на r4uab.ru.")
except requests.exceptions.RequestException as e:
    print(f"Ошибка при запросе данных телеметрии с r4uab.ru: {e}")
except Exception as e:
    print(f"Произошла ошибка при парсинге телеметрии с r4uab.ru: {e}")

# --- 0. КОНСТАНТЫ И ПРЕДПОЛОЖЕНИЯ ---
ts = load.timescale()
print("Загрузка эфемерид Skyfield...")
eph = load('de421.bsp');
earth = eph['earth'];
sun = eph['sun']
print("Эфемериды загружены.")
time_utc = None
if Date_TLM and Time_value:  # Используем время из телеметрии BS4, если оно есть
    try:
        clean_date_str = Date_TLM.replace(" г.", "").strip()
        clean_time_str = Time_value.replace(" UTC", "").strip()
        dt_obj = dt_datetime.strptime(f"{clean_date_str} {clean_time_str}", "%d.%m.%Y %H:%M:%S")
        time_utc = ts.utc(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute, dt_obj.second)
        print(f"Используется время из телеметрии r4uab.ru: {time_utc.utc_iso()}")
    except (ValueError, TypeError) as e:  # Добавил 'as e' для отладки
        print(f"ОШИБКА при парсинге времени из телеметрии: {e}")  # Печатаем ошибку
        time_utc = None  # Убедимся, что time_utc сброшен в случае ошибки

if time_utc is None:  # Если BS4 не дал времени, используем текущее
    current_time_for_calc = ts.now()
    print(
        f"ПРЕДУПРЕЖДЕНИЕ: Не удалось установить время из телеметрии. Используется ТЕКУЩЕЕ ВРЕМЯ UTC: {current_time_for_calc.utc_iso()}")
    time_utc = current_time_for_calc

# --- ПОЛУЧЕНИЕ КООРДИНАТ КА ЧЕРЕЗ PLAYWRIGHT ---
print("\n--- Автоматическое получение координат КА с r4uab.ru (через iframe)... ---")
# Запускаем асинхронную функцию Playwright
loop = asyncio.get_event_loop()
if loop.is_closed():  # Для сред, где цикл может быть закрыт (например, Jupyter notebook после перезапуска ядра)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

try:
    sat_lat_deg_pw, sat_lon_deg_pw, sat_alt_km_pw = loop.run_until_complete(get_coordinates_with_playwright())
except RuntimeError as e_loop:  # Если цикл уже запущен (например, в Jupyter)
    if " asyncio.run() cannot be called from a running event loop" in str(e_loop):
        print("Playwright: Обнаружен уже запущенный цикл событий. Попытка альтернативного запуска...")
        sat_lat_deg_pw, sat_lon_deg_pw, sat_alt_km_pw = None, None, None  # Флаг, что не сработало
        print("Playwright: Альтернативный запуск не реализован, потребуется ручной ввод координат.")
    else:
        raise e_loop

if sat_lat_deg_pw is not None and sat_lon_deg_pw is not None and sat_alt_km_pw is not None:
    sat_lat_deg = sat_lat_deg_pw
    sat_lon_deg = sat_lon_deg_pw
    sat_alt_km = sat_alt_km_pw
    print(f"Координаты КА получены автоматически: Lat={sat_lat_deg:.3f}, Lon={sat_lon_deg:.3f}, Alt={sat_alt_km:.3f}")
else:
    print("ПРЕДУПРЕЖДЕНИЕ: Не удалось автоматически получить координаты КА. Запрошен ручной ввод.")


    def get_float_input(prompt_text, default_val_str=None):  # Определяем снова, если не было определено
        while True:
            try:
                return float(
                    input(f"{prompt_text} (например: {default_val_str if default_val_str else 'число'}): ").replace(',',
                                                                                                                    '.'))
            except ValueError:
                print("Ошибка: Введите корректное число.")


    sat_lat_deg = get_float_input("Широта (градусы, южная - отрицательная)", "-44.674")
    sat_lon_deg = get_float_input("Долгота (градусы, восточная)", "349.343")
    sat_alt_km = get_float_input("Высота (км)", "312.662")

def parse_current_value(current_str, default_val_amps=0.001):  # Переименовал для ясности
    if current_str is None: return default_val_amps
    current_str = current_str.strip().lower();
    val = 0.0
    try:
        match = re.search(r"([-+]?\d*\.?\d+|\d+\.?\d*)", current_str)
        if match:
            val = float(match.group(1))
            if "ма" in current_str or "мa" in current_str:
                val /= 1000.0
            elif "a" not in current_str and "а" not in current_str and abs(val) > 1.0 and abs(val) < 2000:
                val /= 1000.0
        else:
            return default_val_amps
    except ValueError:
        return default_val_amps
    return val


I_sb_current = {1: parse_current_value(electricity1, 0.001), 2: parse_current_value(electricity2, 0.001),
                3: parse_current_value(electricity3, 0.001)}
print(f"Используемые токи СБ (А): {I_sb_current}")
I_max_panel = {1: 0.15, 2: 0.15, 3: 0.15}
print(f"Используемые I_max СБ (А): {I_max_panel}")
satellite_dimensions = np.array([0.1, 0.1, 0.2])
NADIR_POINTING_AXIS_BODY = np.array([0.0, 0.0, 1.0])
print(f"Предположение для надирного наведения: ось {NADIR_POINTING_AXIS_BODY} в СК КА направлена на надир.")
default_n_panel_body_for_calc = {
    1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0]), 3: np.array([0.0, 0.0, 1.0])}
print(f"Используемые нормали панелей (конфигурация по умолчанию): {default_n_panel_body_for_calc}")

satellite_sky_obj = earth + wgs84.latlon(latitude_degrees=sat_lat_deg, longitude_degrees=sat_lon_deg,
                                         elevation_m=sat_alt_km * 1000)


# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def solve_sun_vector_body(I_sb_current_vals, I_max_panel_vals, n_panels_body_map_vals):
    cos_alphas = {}
    for panel_id, n_vec in n_panels_body_map_vals.items():
        if panel_id in I_sb_current_vals and panel_id in I_max_panel_vals:
            if I_max_panel_vals[panel_id] < 1e-6:
                cos_alpha = 0.0
            else:
                cos_alpha = min(max(I_sb_current_vals[panel_id] / I_max_panel_vals[panel_id], -1.0), 1.0)
            cos_alphas[panel_id] = cos_alpha
        else:
            cos_alphas[panel_id] = 0.0
    s_sun_body_unnormalized = np.zeros(3);
    num_active_panels_for_sum = 0
    for panel_id, n_vec_panel in n_panels_body_map_vals.items():
        if panel_id in cos_alphas:
            s_sun_body_unnormalized += n_vec_panel * cos_alphas[panel_id]
            if abs(cos_alphas[panel_id]) > 1e-3: num_active_panels_for_sum += 1
    if num_active_panels_for_sum == 0: return np.array([1.0, 0.0, 0.0])
    norm = np.linalg.norm(s_sun_body_unnormalized)
    if norm < 1e-9:
        if all(abs(I_sb_current_vals.get(pid, 0)) < 1e-3 for pid in n_panels_body_map_vals): pass
        return np.array([1.0, 0.0, 0.0])
    return s_sun_body_unnormalized / norm


def get_sun_vector_global(satellite_object, time_val):
    astrometric = satellite_object.at(time_val).observe(sun)
    s_sun_global_unnormalized = astrometric.position.km
    if np.linalg.norm(s_sun_global_unnormalized) < 1e-9: return np.array([1.0, 0.0, 0.0])
    return s_sun_global_unnormalized / np.linalg.norm(s_sun_global_unnormalized)


def get_nadir_vector_global(satellite_object, time_val):
    sat_pos_gcrs = satellite_object.at(time_val).position.km
    nadir_global_unnormalized = -sat_pos_gcrs
    if np.linalg.norm(nadir_global_unnormalized) < 1e-9: return np.array([0.0, 0.0, -1.0])
    return nadir_global_unnormalized / np.linalg.norm(nadir_global_unnormalized)


def calculate_attitude_triad(v1_body_vec, v2_body_vec, v1_global_vec, v2_global_vec):
    for vec_name, vec in [("v1_body", v1_body_vec), ("v2_body", v2_body_vec), ("v1_global", v1_global_vec),
                          ("v2_global", v2_global_vec)]:
        if np.linalg.norm(vec) < 1e-9: raise ValueError(f"Вектор {vec_name} для TRIAD имеет нулевую длину: {vec}")
    v1_b = v1_body_vec / np.linalg.norm(v1_body_vec);
    v2_b = v2_body_vec / np.linalg.norm(v2_body_vec)
    v1_g = v1_global_vec / np.linalg.norm(v1_global_vec);
    v2_g = v2_global_vec / np.linalg.norm(v2_global_vec)
    if np.abs(np.dot(v1_b, v2_b)) > 0.999: raise ValueError("Векторы в СК КА (v1_body, v2_body) коллинеарны.")
    if np.abs(np.dot(v1_g, v2_g)) > 0.999: raise ValueError(
        "Векторы в глобальной СК (v1_global, v2_global) коллинеарны.")
    t1_b = v1_b;
    t2_b_unnorm = np.cross(v1_b, v2_b)
    if np.linalg.norm(t2_b_unnorm) < 1e-9: raise ValueError("np.cross(v1_b, v2_b) дал нулевой вектор.")
    t2_b = t2_b_unnorm / np.linalg.norm(t2_b_unnorm);
    t3_b = np.cross(t1_b, t2_b)
    M_body = np.array([t1_b, t2_b, t3_b]).T
    t1_g = v1_g;
    t2_g_unnorm = np.cross(v1_g, v2_g)
    if np.linalg.norm(t2_g_unnorm) < 1e-9: raise ValueError("np.cross(v1_g, v2_g) дал нулевой вектор.")
    t2_g = t2_g_unnorm / np.linalg.norm(t2_g_unnorm);
    t3_g = np.cross(t1_g, t2_g)
    M_global = np.array([t1_g, t2_g, t3_g]).T
    rotation_matrix = np.dot(M_global, M_body.T)
    return rotation_matrix


# --- 3. ОСНОВНЫЕ ВЫЧИСЛЕНИЯ ---
print(f"\nВремя UTC для расчетов: {time_utc.utc_iso()}")
sat_pos_gcrs_km = satellite_sky_obj.at(time_utc).position.km
s_sun_global = get_sun_vector_global(satellite_sky_obj, time_utc)
nadir_global = get_nadir_vector_global(satellite_sky_obj,
                                       time_utc)  # Этот вектор все еще нужен для TRIAD в текущей логике
nadir_body_assumed = NADIR_POINTING_AXIS_BODY / np.linalg.norm(NADIR_POINTING_AXIS_BODY)

print("\n--- Поиск наилучшей гипотезы расположения солнечных панелей ---")
possible_normals_map = {
    "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0])}
face_names_ordered = list(possible_normals_map.keys())
panel_ids_ordered = sorted(list(I_sb_current.keys()))
best_hypothesis_info = None;
max_consistency_score = -2.0;
num_hypotheses = 0
if len(panel_ids_ordered) > 0 and len(face_names_ordered) >= len(panel_ids_ordered):
    for face_indices_for_panels in itertools.permutations(range(len(face_names_ordered)), len(panel_ids_ordered)):
        num_hypotheses += 1
        current_n_panel_body_for_calc_hyp = {}
        hypothesis_desc_list = []
        for i, panel_id in enumerate(panel_ids_ordered):
            face_name = face_names_ordered[face_indices_for_panels[i]]
            current_n_panel_body_for_calc_hyp[panel_id] = possible_normals_map[face_name]
            hypothesis_desc_list.append(f"СБ{panel_id} на {face_name}")
        hypothesis_name_str = "; ".join(hypothesis_desc_list)
        try:
            s_sun_body_hyp = solve_sun_vector_body(I_sb_current, I_max_panel,
                                                   current_n_panel_body_for_calc_hyp)
            if np.linalg.norm(s_sun_body_hyp) < 1e-6: continue
            attitude_matrix_hyp = calculate_attitude_triad(s_sun_body_hyp, nadir_body_assumed, s_sun_global,
                                                           nadir_global)
            s_sun_global_transformed_to_body = attitude_matrix_hyp.T @ s_sun_global
            if np.linalg.norm(s_sun_global_transformed_to_body) > 1e-6:
                s_sun_global_transformed_to_body /= np.linalg.norm(s_sun_global_transformed_to_body)
            else:
                continue
            consistency_score = np.dot(s_sun_body_hyp, s_sun_global_transformed_to_body)
            if consistency_score > max_consistency_score:
                max_consistency_score = consistency_score
                best_hypothesis_info = {'name': hypothesis_name_str, 'config': current_n_panel_body_for_calc_hyp,
                                        'attitude': attitude_matrix_hyp, 's_sun_body': s_sun_body_hyp,
                                        'score': consistency_score}
        except ValueError:
            pass
        except Exception:
            pass
print(f"Проверено гипотез: {num_hypotheses}")

attitude_matrix = np.identity(3);
s_sun_body = np.array([1.0, 0.0, 0.0])
n_panel_body_for_calc_final = default_n_panel_body_for_calc
if best_hypothesis_info:
    print(
        f"\n--- Найдена наилучшая гипотеза: {best_hypothesis_info['name']} (Оценка: {best_hypothesis_info['score']:.4f}) ---")
    attitude_matrix = best_hypothesis_info['attitude'];
    s_sun_body = best_hypothesis_info['s_sun_body']
    n_panel_body_for_calc_final = best_hypothesis_info['config']
else:
    print("\nПРЕДУПРЕЖДЕНИЕ: Не найдена лучшая гипотеза. Используется конфигурация по умолчанию.")
    s_sun_body = solve_sun_vector_body(I_sb_current, I_max_panel,
                                       n_panel_body_for_calc_final)
    try:
        attitude_matrix = calculate_attitude_triad(s_sun_body, nadir_body_assumed, s_sun_global, nadir_global)
    except ValueError as e:
        print(f"Ошибка TRIAD с дефолтной конфигурацией: {e}")

print(f"\n--- Финальные параметры ориентации ---")
print(f"Позиция КА в GCRS [км]: {sat_pos_gcrs_km}")
print(f"Вектор на Солнце в глоб. СК (от КА): {s_sun_global}")
print(f"Вектор на надир в глоб. СК (от КА): {nadir_global}")
print(f"Вектор на Солнце в СК КА: {s_sun_body}")
print(f"Вектор на надир в СК КА (предположение): {nadir_body_assumed}")
print(f"Матрица ориентации (body-to-global):\n{attitude_matrix}")
if not np.allclose(np.dot(attitude_matrix, attitude_matrix.T), np.identity(3)): print(
    "ПРЕДУПРЕЖДЕНИЕ: Матрица не ортонормирована!")
if not np.isclose(np.linalg.det(attitude_matrix), 1.0): print(
    f"ПРЕДУПРЕЖДЕНИЕ: Определитель != 1 (det={np.linalg.det(attitude_matrix)})!")
orientation_quat_scipy = R.from_matrix(attitude_matrix).as_quat()
orientation_euler_xyz_deg = R.from_matrix(attitude_matrix).as_euler('xyz', degrees=True)
print(f"Ориентация (кватернион XYZW, body-to-global): {orientation_quat_scipy}")
print(
    f"Ориентация (углы Эйлера 'xyz', body-to-global) [градусы]: Roll={orientation_euler_xyz_deg[0]:.2f}°, Pitch={orientation_euler_xyz_deg[1]:.2f}°, Yaw={orientation_euler_xyz_deg[2]:.2f}°")

print("\n--- Углы к Солнцу и Надиру в СК КА ---")
if np.linalg.norm(s_sun_body) > 1e-6:
    az_sun_body = np.degrees(np.arctan2(s_sun_body[1], s_sun_body[0]))
    el_sun_body = np.degrees(np.arcsin(np.clip(s_sun_body[2], -1, 1)))
    print(f"  Солнце в СК КА: Азимут={az_sun_body:.2f}°, Элевация={el_sun_body:.2f}°")
if np.linalg.norm(nadir_body_assumed) > 1e-6:
    az_nadir_body = np.degrees(np.arctan2(nadir_body_assumed[1], nadir_body_assumed[0]))
    el_nadir_body = np.degrees(np.arcsin(np.clip(nadir_body_assumed[2], -1, 1)))
    print(f"  Надир в СК КА (по предположению): Азимут={az_nadir_body:.2f}°, Элевация={el_nadir_body:.2f}°")

plotter = pv.Plotter(window_size=[1000, 600], lighting="light_kit")
plotter.background_color = 'black'
satellite_body_mesh_local = pv.Box(
    bounds=[-satellite_dimensions[0] / 2, satellite_dimensions[0] / 2, -satellite_dimensions[1] / 2,
            satellite_dimensions[1] / 2, -satellite_dimensions[2] / 2, satellite_dimensions[2] / 2, ])
body_actor = plotter.add_mesh(satellite_body_mesh_local.copy(), color='red', edge_color='blue', show_edges=True,
                              line_width=3, specular=0.6, specular_power=10)
rotation_transform = np.eye(4);
rotation_transform[:3, :3] = attitude_matrix
body_actor.user_matrix = rotation_transform
axis_length = np.max(satellite_dimensions) * 1.5;
origin_body = np.array([0.0, 0.0, 0.0])
x_axis_body = np.array([axis_length, 0, 0]);
y_axis_body = np.array([0, axis_length, 0]);
z_axis_body = np.array([0, 0, axis_length])
x_axis_global_vis = attitude_matrix @ x_axis_body;
y_axis_global_vis = attitude_matrix @ y_axis_body;
z_axis_global_vis = attitude_matrix @ z_axis_body
plotter.add_arrows(cent=origin_body, direction=x_axis_global_vis, mag=1, color='darkred', line_width=3, label="X_КА")
plotter.add_arrows(cent=origin_body, direction=y_axis_global_vis, mag=1, color='darkgreen', line_width=3, label="Y_КА")
plotter.add_arrows(cent=origin_body, direction=z_axis_global_vis, mag=1, color='darkblue', line_width=3, label="Z_КА")
if n_panel_body_for_calc_final:  # Отрисовка панелей
    panel_thickness_ratio = 0.05;
    panel_offset_ratio = 0.01
    face_plane_axes_map = {(1, 0, 0): [1, 2], (-1, 0, 0): [1, 2], (0, 1, 0): [0, 2], (0, -1, 0): [0, 2],
                           (0, 0, 1): [0, 1], (0, 0, -1): [0, 1], }
    for panel_id, normal_vec_panel in n_panel_body_for_calc_final.items():
        panel_color = 'deepskyblue'
        normal_tuple = tuple(int(x) for x in np.sign(normal_vec_panel))
        if normal_tuple not in face_plane_axes_map: continue
        dims_axes_indices = face_plane_axes_map[normal_tuple]
        thickness_axis_idx = np.where(np.abs(normal_vec_panel) > 0.5)[0][0]
        panel_dim_local = np.zeros(3)
        panel_dim_local[thickness_axis_idx] = satellite_dimensions[thickness_axis_idx] * panel_thickness_ratio
        dim_idx1 = dims_axes_indices[0];
        dim_idx2 = dims_axes_indices[1]
        panel_dim_local[dim_idx1] = satellite_dimensions[dim_idx1] * 0.95
        panel_dim_local[dim_idx2] = satellite_dimensions[dim_idx2] * 0.95
        panel_mesh_local = pv.Box(
            bounds=[-panel_dim_local[0] / 2, panel_dim_local[0] / 2, -panel_dim_local[1] / 2, panel_dim_local[1] / 2,
                    -panel_dim_local[2] / 2, panel_dim_local[2] / 2, ])
        panel_center_offset_local = normal_vec_panel * (
                    satellite_dimensions[thickness_axis_idx] / 2 * (1 + panel_offset_ratio))
        panel_mesh_local.translate(panel_center_offset_local, inplace=True)
        panel_actor = plotter.add_mesh(panel_mesh_local.copy(), color=panel_color, show_edges=True, ambient=0.3,
                                       diffuse=0.7)
        panel_actor.user_matrix = rotation_transform
symbolic_dist_factor = np.max(satellite_dimensions) * 8
earth_symbol_position = np.array([symbolic_dist_factor, 0, 0])
earth_marker = pv.Sphere(radius=np.max(satellite_dimensions) * 0.7, center=earth_symbol_position)
plotter.add_mesh(earth_marker, color='green', label="Земля (символ)")
arrow_scale_directions = np.max(satellite_dimensions) * 5
plotter.add_arrows(cent=origin_body, direction=attitude_matrix @ nadir_body_assumed, mag=arrow_scale_directions,
                   color='purple', line_width=5, label="Напр. к Земле (по оси КА)")
plotter.add_arrows(cent=origin_body, direction=attitude_matrix @ s_sun_body, mag=arrow_scale_directions, color='yellow',
                   line_width=5, label="Напр. к Солнцу (по осям КА)")
plotter.add_legend(
    labels=[("X_КА", "darkred"), ("Y_КА", "darkgreen"), ("Z_КА", "darkblue"), ("Земля (символ)", "green"),
            ("Напр. к Земле (по оси КА)", "purple"), ("Напр. к Солнцу (по осям КА)", "yellow")], bcolor=None,
    face='none', border=True, size=(0.25, 0.22))
plotter.camera.focal_point = [0, 0, 0];
cam_pos_factor = np.max(satellite_dimensions) * 15
plotter.camera.position = [-cam_pos_factor * 0.8, -cam_pos_factor, cam_pos_factor * 0.5]
plotter.camera.view_up = [0, 0, 1];
plotter.camera.zoom(1.0)
plotter.enable_anti_aliasing('fxaa')
print("\nИнструкции для PyVista окна: (ЛКМ - вращение, колесико/ПКМ - зум, СКМ - панорама)")
plotter.show()
print("\nСкрипт завершен.")
