"""
API-маршруты для симуляции
Step2: Чтение и фильтрация сущностей Zep, подготовка и запуск симуляции OASIS (полная автоматизация)
"""

import os
import traceback
from flask import request, jsonify, send_file

from . import simulation_bp
from ..config import Config
from ..services.zep_entity_reader import ZepEntityReader
from ..services.oasis_profile_generator import OasisProfileGenerator
from ..services.simulation_manager import SimulationManager, SimulationStatus
from ..services.simulation_runner import SimulationRunner, RunnerStatus
from ..utils.logger import get_logger
from ..models.project import ProjectManager

logger = get_logger('mirofish.api.simulation')


# Оптимизирующий префикс для Interview prompt
# Добавление этого префикса не даёт Agent вызывать инструменты, заставляя отвечать текстом
INTERVIEW_PROMPT_PREFIX = "Основываясь на своём персонаже, всех прошлых воспоминаниях и действиях, ответь мне текстом напрямую, не вызывая никаких инструментов:"


def optimize_interview_prompt(prompt: str) -> str:
    """
    Оптимизация вопроса Interview, добавление префикса для предотвращения вызова инструментов Agent
    
    Args:
        prompt: Исходный вопрос
        
    Returns:
        Оптимизированный вопрос
    """
    if not prompt:
        return prompt
    # Предотвращение повторного добавления префикса
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


# ============== Интерфейс чтения сущностей ==============

@simulation_bp.route('/entities/<graph_id>', methods=['GET'])
def get_graph_entities(graph_id: str):
    """
    Получение всех сущностей графа (отфильтрованных)
    
    Возвращает только узлы, соответствующие предопределённым типам сущностей (узлы с Labels не только Entity)
    
    Query-параметры:
        entity_types: Список типов сущностей через запятую (необязательно, для дополнительной фильтрации)
        enrich: Получать ли информацию о связанных рёбрах (по умолчанию true)
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY не настроен"
            }), 500
        
        entity_types_str = request.args.get('entity_types', '')
        entity_types = [t.strip() for t in entity_types_str.split(',') if t.strip()] if entity_types_str else None
        enrich = request.args.get('enrich', 'true').lower() == 'true'
        
        logger.info(f"Получение сущностей графа: graph_id={graph_id}, entity_types={entity_types}, enrich={enrich}")
        
        reader = ZepEntityReader()
        result = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=enrich
        )
        
        return jsonify({
            "success": True,
            "data": result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения сущностей графа: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/<entity_uuid>', methods=['GET'])
def get_entity_detail(graph_id: str, entity_uuid: str):
    """Получение детальной информации о сущности"""
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY не настроен"
            }), 500
        
        reader = ZepEntityReader()
        entity = reader.get_entity_with_context(graph_id, entity_uuid)
        
        if not entity:
            return jsonify({
                "success": False,
                "error": f"Сущность не найдена: {entity_uuid}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": entity.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения деталей сущности: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/by-type/<entity_type>', methods=['GET'])
def get_entities_by_type(graph_id: str, entity_type: str):
    """Получение всех сущностей указанного типа"""
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY не настроен"
            }), 500
        
        enrich = request.args.get('enrich', 'true').lower() == 'true'
        
        reader = ZepEntityReader()
        entities = reader.get_entities_by_type(
            graph_id=graph_id,
            entity_type=entity_type,
            enrich_with_edges=enrich
        )
        
        return jsonify({
            "success": True,
            "data": {
                "entity_type": entity_type,
                "count": len(entities),
                "entities": [e.to_dict() for e in entities]
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения сущностей: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Интерфейс управления симуляцией ==============

@simulation_bp.route('/create', methods=['POST'])
def create_simulation():
    """
    Создание новой симуляции
    
    Примечание: параметры max_rounds и др. генерируются LLM автоматически, ручная настройка не требуется
    
    Запрос (JSON):
        {
            "project_id": "proj_xxxx",      // обязательно
            "graph_id": "mirofish_xxxx",    // необязательно, если не указан — берётся из project
            "enable_twitter": true,          // необязательно, по умолчанию true
            "enable_reddit": true            // необязательно, по умолчанию true
        }
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        project_id = data.get('project_id')
        if not project_id:
            return jsonify({
                "success": False,
                "error": "Укажите project_id"
            }), 400
        
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Проект не найден: {project_id}"
            }), 404
        
        graph_id = data.get('graph_id') or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Граф проекта ещё не построен, сначала вызовите /api/graph/build"
            }), 400
        
        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get('enable_twitter', True),
            enable_reddit=data.get('enable_reddit', True),
        )
        
        return jsonify({
            "success": True,
            "data": state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Ошибка создания симуляции: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Проверка готовности симуляции
    
    Условия проверки:
    1. state.json существует и status = "ready"
    2. Необходимые файлы существуют: reddit_profiles.json, twitter_profiles.csv, simulation_config.json
    
    Примечание: скрипты запуска (run_*.py) остаются в backend/scripts/, не копируются в директорию симуляции
    
    Args:
        simulation_id: ID симуляции
        
    Returns:
        (is_prepared: bool, info: dict)
    """
    import os
    from ..config import Config
    
    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
    
    # Проверка существования директории
    if not os.path.exists(simulation_dir):
        return False, {"reason": "Директория симуляции не существует"}
    
    # Список необходимых файлов (без скриптов, скрипты находятся в backend/scripts/)
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv"
    ]
    
    # Проверка существования файлов
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)
    
    if missing_files:
        return False, {
            "reason": "Отсутствуют необходимые файлы",
            "missing_files": missing_files,
            "existing_files": existing_files
        }
    
    # Проверка статуса в state.json
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        import json
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)
        
        # Подробное логирование
        logger.debug(f"Проверка статуса подготовки симуляции: {simulation_id}, status={status}, config_generated={config_generated}")
        
        # Если config_generated=True и файлы существуют, считаем подготовку завершённой
        # Следующие статусы означают завершение подготовки:
        # - ready: подготовка завершена, можно запускать
        # - preparing: если config_generated=True — уже завершено
        # - running: запущено, подготовка давно завершена
        # - completed: выполнение завершено, подготовка давно завершена
        # - stopped: остановлено, подготовка давно завершена
        # - failed: выполнение провалилось (но подготовка была завершена)
        prepared_statuses = ["ready", "preparing", "running", "completed", "stopped", "failed"]
        if status in prepared_statuses and config_generated:
            # Получение статистики файлов
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            config_file = os.path.join(simulation_dir, "simulation_config.json")
            
            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    profiles_count = len(profiles_data) if isinstance(profiles_data, list) else 0
            
            # Если статус preparing, но файлы готовы — автоматически обновляем статус на ready
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    from datetime import datetime
                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Автообновление статуса симуляции: {simulation_id} preparing -> ready")
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Ошибка автообновления статуса: {e}")
            
            logger.info(f"Симуляция {simulation_id} результат проверки: подготовка завершена (status={status}, config_generated={config_generated})")
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files
            }
        else:
            logger.warning(f"Симуляция {simulation_id} результат проверки: подготовка не завершена (status={status}, config_generated={config_generated})")
            return False, {
                "reason": f"Статус не в списке готовых или config_generated=false: status={status}, config_generated={config_generated}",
                "status": status,
                "config_generated": config_generated
            }
            
    except Exception as e:
        return False, {"reason": f"Ошибка чтения файла состояния: {str(e)}"}


@simulation_bp.route('/prepare', methods=['POST'])
def prepare_simulation():
    """
    Подготовка среды симуляции (асинхронная задача, LLM автоматически генерирует все параметры)
    
    Длительная операция, интерфейс сразу возвращает task_id,
    используйте GET /api/simulation/prepare/status для отслеживания прогресса
    
    Особенности:
    - Автоматическое обнаружение завершённой подготовки, предотвращение повторной генерации
    - Если подготовка завершена — сразу возвращает существующий результат
    - Поддержка принудительной регенерации (force_regenerate=true)
    
    Шаги:
    1. Проверка наличия завершённой подготовки
    2. Чтение и фильтрация сущностей из графа Zep
    3. Генерация OASIS Agent Profile для каждой сущности (с механизмом повторных попыток)
    4. Автоматическая генерация конфигурации симуляции LLM (с механизмом повторных попыток)
    5. Сохранение конфигурационных файлов и скриптов
    
    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",                   // обязательно, ID симуляции
            "entity_types": ["Student", "PublicFigure"],  // необязательно, указание типов сущностей
            "use_llm_for_profiles": true,                 // необязательно, генерировать ли профили через LLM
            "parallel_profile_count": 5,                  // необязательно, кол-во параллельной генерации профилей, по умолчанию 5
            "force_regenerate": false                     // необязательно, принудительная регенерация, по умолчанию false
        }
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // возвращается для новой задачи
                "status": "preparing|ready",
                "message": "Задача подготовки запущена|Подготовка уже завершена",
                "already_prepared": true|false    // завершена ли подготовка
            }
        }
    """
    import threading
    import os
    from ..models.task import TaskManager, TaskStatus
    from ..config import Config
    
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400
        
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": f"Симуляция не найдена: {simulation_id}"
            }), 404
        
        # Проверка принудительной регенерации
        force_regenerate = data.get('force_regenerate', False)
        logger.info(f"Начало обработки /prepare запроса: simulation_id={simulation_id}, force_regenerate={force_regenerate}")
        
        # Проверка завершённости подготовки (предотвращение повторной генерации)
        if not force_regenerate:
            logger.debug(f"Проверка готовности симуляции {simulation_id}...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(f"Результат проверки: is_prepared={is_prepared}, prepare_info={prepare_info}")
            if is_prepared:
                logger.info(f"Симуляция {simulation_id} уже подготовлена, пропуск повторной генерации")
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": "Подготовка уже завершена, повторная генерация не требуется",
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
            else:
                logger.info(f"Симуляция {simulation_id} не подготовлена, запуск задачи подготовки")
        
        # Получение необходимой информации из проекта
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Проект не найден: {state.project_id}"
            }), 404
        
        # Получение требований симуляции
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "В проекте отсутствует описание требований симуляции (simulation_requirement)"
            }), 400
        
        # Получение текста документа
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""
        
        entity_types_list = data.get('entity_types')
        use_llm_for_profiles = data.get('use_llm_for_profiles', True)
        parallel_profile_count = data.get('parallel_profile_count', 5)
        
        # ========== Синхронное получение кол-ва сущностей (до запуска фоновой задачи) ==========
        # Чтобы фронтенд сразу после вызова prepare получил ожидаемое кол-во Agent
        try:
            logger.info(f"Синхронное получение кол-ва сущностей: graph_id={state.graph_id}")
            reader = ZepEntityReader()
            # Быстрое чтение сущностей (без информации о рёбрах, только подсчёт)
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False  # Без информации о рёбрах, для ускорения
            )
            # Сохранение кол-ва сущностей в состояние (для немедленного получения фронтендом)
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(f"Ожидаемое кол-во сущностей: {filtered_preview.filtered_count}, типы: {filtered_preview.entity_types}")
        except Exception as e:
            logger.warning(f"Ошибка синхронного получения кол-ва сущностей (повтор в фоновой задаче): {e}")
            # Ошибка не влияет на дальнейший процесс, фоновая задача повторит попытку
        
        # Создание асинхронной задачи
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={
                "simulation_id": simulation_id,
                "project_id": state.project_id
            }
        )
        
        # Обновление состояния симуляции (с предварительно полученным кол-вом сущностей)
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)
        
        # Определение фоновой задачи
        def run_prepare():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Начало подготовки среды симуляции..."
                )
                
                # Подготовка симуляции (с callback прогресса)
                # Хранение деталей прогресса этапов
                stage_details = {}
                
                def progress_callback(stage, progress, message, **kwargs):
                    # Расчёт общего прогресса
                    stage_weights = {
                        "reading": (0, 20),           # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),    # 70-90%
                        "copying_scripts": (90, 100)       # 90-100%
                    }
                    
                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)
                    
                    # Построение детальной информации о прогрессе
                    stage_names = {
                        "reading": "Чтение сущностей графа",
                        "generating_profiles": "Генерация профилей Agent",
                        "generating_config": "Генерация конфигурации симуляции",
                        "copying_scripts": "Подготовка скриптов симуляции"
                    }
                    
                    stage_index = list(stage_weights.keys()).index(stage) + 1 if stage in stage_weights else 1
                    total_stages = len(stage_weights)
                    
                    # Обновление деталей этапа
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", "")
                    }
                    
                    # Построение детальной информации о прогрессе
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message
                    }
                    
                    # Построение краткого сообщения
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"
                    
                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data
                    )
                
                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count
                )
                
                # Задача завершена
                task_manager.complete_task(
                    task_id,
                    result=result_state.to_simple_dict()
                )
                
            except Exception as e:
                logger.error(f"Ошибка подготовки симуляции: {str(e)}")
                task_manager.fail_task(task_id, str(e))
                
                # Обновление статуса симуляции на failed
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)
        
        # Запуск фонового потока
        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": "Задача подготовки запущена, отслеживайте прогресс через /api/simulation/prepare/status",
                "already_prepared": False,
                "expected_entities_count": state.entities_count,  # Ожидаемое общее кол-во Agent
                "entity_types": state.entity_types  # Список типов сущностей
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Ошибка запуска задачи подготовки: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/prepare/status', methods=['POST'])
def get_prepare_status():
    """
    Запрос прогресса задачи подготовки
    
    Поддерживает два способа запроса:
    1. По task_id — прогресс текущей задачи
    2. По simulation_id — проверка наличия завершённой подготовки
    
    Запрос (JSON):
        {
            "task_id": "task_xxxx",          // необязательно, task_id из prepare
            "simulation_id": "sim_xxxx"      // необязательно, ID симуляции (для проверки готовности)
        }
    
    Ответ:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // наличие завершённой подготовки
                "prepare_info": {...}            // детали при завершённой подготовке
            }
        }
    """
    from ..models.task import TaskManager
    
    try:
        data = request.get_json() or {}
        
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # Если предоставлен simulation_id, сначала проверяем готовность
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "progress": 100,
                        "message": "Подготовка уже завершена",
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
        
        # Если нет task_id — возвращаем ошибку
        if not task_id:
            if simulation_id:
                # Есть simulation_id, но подготовка не завершена
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "not_started",
                        "progress": 0,
                        "message": "Подготовка ещё не начата, вызовите /api/simulation/prepare",
                        "already_prepared": False
                    }
                })
            return jsonify({
                "success": False,
                "error": "Укажите task_id или simulation_id"
            }), 400
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            # Задача не найдена, но если есть simulation_id — проверяем готовность
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify({
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "task_id": task_id,
                            "status": "ready",
                            "progress": 100,
                            "message": "Задача завершена (подготовка уже существует)",
                            "already_prepared": True,
                            "prepare_info": prepare_info
                        }
                    })
            
            return jsonify({
                "success": False,
                "error": f"Задача не найдена: {task_id}"
            }), 404
        
        task_dict = task.to_dict()
        task_dict["already_prepared"] = False
        
        return jsonify({
            "success": True,
            "data": task_dict
        })
        
    except Exception as e:
        logger.error(f"Ошибка запроса статуса задачи: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@simulation_bp.route('/<simulation_id>', methods=['GET'])
def get_simulation(simulation_id: str):
    """Получение статуса симуляции"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": f"Симуляция не найдена: {simulation_id}"
            }), 404
        
        result = state.to_dict()
        
        # Если симуляция готова, добавляем инструкции запуска
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения статуса симуляции: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/list', methods=['GET'])
def list_simulations():
    """
    Список всех симуляций
    
    Query-параметры:
        project_id: Фильтр по ID проекта (необязательно)
    """
    try:
        project_id = request.args.get('project_id')
        
        manager = SimulationManager()
        simulations = manager.list_simulations(project_id=project_id)
        
        return jsonify({
            "success": True,
            "data": [s.to_dict() for s in simulations],
            "count": len(simulations)
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения списка симуляций: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _get_report_id_for_simulation(simulation_id: str) -> str:
    """
    Получение последнего report_id для simulation
    
    Обход директории reports, поиск report с совпадающим simulation_id,
    если несколько — возвращается самый новый (по created_at)
    
    Args:
        simulation_id: ID симуляции
        
    Returns:
        report_id или None
    """
    import json
    from datetime import datetime
    
    # Путь к директории reports: backend/uploads/reports
    # __file__ — это app/api/simulation.py, нужно подняться на 2 уровня до backend/
    reports_dir = os.path.join(os.path.dirname(__file__), '../../uploads/reports')
    if not os.path.exists(reports_dir):
        return None
    
    matching_reports = []
    
    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue
            
            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue
            
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append({
                        "report_id": meta.get("report_id"),
                        "created_at": meta.get("created_at", ""),
                        "status": meta.get("status", "")
                    })
            except Exception:
                continue
        
        if not matching_reports:
            return None
        
        # Сортировка по времени создания (по убыванию), возврат самого нового
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")
        
    except Exception as e:
        logger.warning(f"Ошибка поиска report для simulation {simulation_id}: {e}")
        return None


@simulation_bp.route('/history', methods=['GET'])
def get_simulation_history():
    """
    Получение истории симуляций (с деталями проектов)
    
    Для отображения истории на главной странице, возвращает список симуляций с названиями проектов, описаниями и пр.
    
    Query-параметры:
        limit: Лимит количества (по умолчанию 20)
    
    Ответ:
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "Анализ общественного мнения",
                    "simulation_requirement": "Если университет опубликует...",
                    "status": "completed",
                    "entities_count": 68,
                    "profiles_count": 68,
                    "entity_types": ["Student", "Professor", ...],
                    "created_at": "2024-12-10",
                    "updated_at": "2024-12-10",
                    "total_rounds": 120,
                    "current_round": 120,
                    "report_id": "report_xxxx",
                    "version": "v1.0.2"
                },
                ...
            ],
            "count": 7
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        
        manager = SimulationManager()
        simulations = manager.list_simulations()[:limit]
        
        # Обогащение данных симуляции, чтение только из файлов Simulation
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()
            
            # Получение конфигурации симуляции (чтение simulation_requirement из simulation_config.json)
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get("simulation_requirement", "")
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get("total_simulation_hours", 0)
                # Рекомендованное кол-во раундов (запасное значение)
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0) * 60 / 
                    max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0
            
            # Получение статуса выполнения (чтение фактического кол-ва раундов из run_state.json)
            run_state = SimulationRunner.get_run_state(sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Используем пользовательский total_rounds, при отсутствии — рекомендованное кол-во
                sim_dict["total_rounds"] = run_state.total_rounds if run_state.total_rounds > 0 else recommended_rounds
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds
            
            # Получение списка файлов связанного проекта (максимум 3)
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, 'files') and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "Неизвестный файл")} 
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []
            
            # Получение связанного report_id (поиск последнего report для этой simulation)
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)
            
            # Добавление номера версии
            sim_dict["version"] = "v1.0.2"
            
            # Форматирование даты
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""
            
            enriched_simulations.append(sim_dict)
        
        return jsonify({
            "success": True,
            "data": enriched_simulations,
            "count": len(enriched_simulations)
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения истории симуляций: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles', methods=['GET'])
def get_simulation_profiles(simulation_id: str):
    """
    Получение Agent Profile симуляции
    
    Query-параметры:
        platform: Тип платформы (reddit/twitter, по умолчанию reddit)
    """
    try:
        platform = request.args.get('platform', 'reddit')
        
        manager = SimulationManager()
        profiles = manager.get_profiles(simulation_id, platform=platform)
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "count": len(profiles),
                "profiles": profiles
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Ошибка получения Profile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles/realtime', methods=['GET'])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    Получение Agent Profile симуляции в реальном времени (для просмотра прогресса во время генерации)
    
    Отличие от интерфейса /profiles:
    - Прямое чтение файла, без SimulationManager
    - Подходит для просмотра в процессе генерации
    - Возвращает дополнительные метаданные (время изменения файла, идёт ли генерация и т.д.)
    
    Query-параметры:
        platform: Тип платформы (reddit/twitter, по умолчанию reddit)
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // ожидаемое общее кол-во (если есть)
                "is_generating": true,  // идёт ли генерация
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "profiles": [...]
            }
        }
    """
    import json
    import csv
    from datetime import datetime
    
    try:
        platform = request.args.get('platform', 'reddit')
        
        # Получение директории симуляции
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": f"Симуляция не найдена: {simulation_id}"
            }), 404
        
        # Определение пути к файлу
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")
        
        # Проверка существования файлов
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None
        
        if file_exists:
            # Получение времени изменения файла
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                if platform == "reddit":
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        profiles = json.load(f)
                else:
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        profiles = list(reader)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Ошибка чтения файла profiles (возможно, идёт запись): {e}")
                profiles = []
        
        # Проверка идёт ли генерация (по state.json)
        is_generating = False
        total_expected = None
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    total_expected = state_data.get("entities_count")
            except Exception:
                pass
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "platform": platform,
                "count": len(profiles),
                "total_expected": total_expected,
                "is_generating": is_generating,
                "file_exists": file_exists,
                "file_modified_at": file_modified_at,
                "profiles": profiles
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения Profile в реальном времени: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/realtime', methods=['GET'])
def get_simulation_config_realtime(simulation_id: str):
    """
    Получение конфигурации симуляции в реальном времени (для просмотра прогресса во время генерации)
    
    Отличие от интерфейса /config:
    - Прямое чтение файла, без SimulationManager
    - Подходит для просмотра в процессе генерации
    - Возвращает дополнительные метаданные (время изменения файла, идёт ли генерация и т.д.)
    - Возвращает частичную информацию даже если генерация не завершена
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // идёт ли генерация
                "generation_stage": "generating_config",  // текущий этап генерации
                "config": {...}  // содержимое конфигурации (если существует)
            }
        }
    """
    import json
    from datetime import datetime
    
    try:
        # Получение директории симуляции
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": f"Симуляция не найдена: {simulation_id}"
            }), 404
        
        # Путь к файлу конфигурации
        config_file = os.path.join(sim_dir, "simulation_config.json")
        
        # Проверка существования файлов
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None
        
        if file_exists:
            # Получение времени изменения файла
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Ошибка чтения файла config (возможно, идёт запись): {e}")
                config = None
        
        # Проверка идёт ли генерация (по state.json)
        is_generating = False
        generation_stage = None
        config_generated = False
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    config_generated = state_data.get("config_generated", False)
                    
                    # Определение текущего этапа
                    if is_generating:
                        if state_data.get("profiles_generated", False):
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "ready":
                        generation_stage = "completed"
            except Exception:
                pass
        
        # Построение данных ответа
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "config_generated": config_generated,
            "config": config
        }
        
        # Если конфигурация существует — извлекаем ключевую статистику
        if config:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get("total_simulation_hours"),
                "initial_posts_count": len(config.get("event_config", {}).get("initial_posts", [])),
                "hot_topics_count": len(config.get("event_config", {}).get("hot_topics", [])),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model")
            }
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения Config в реальном времени: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config', methods=['GET'])
def get_simulation_config(simulation_id: str):
    """
    Получение конфигурации симуляции (полная конфигурация, сгенерированная LLM)
    
    Возвращает:
        - time_config: настройки времени (длительность симуляции, раунды, пиковые/спадовые периоды)
        - agent_configs: настройки активности каждого Agent (активность, частота сообщений, позиция и т.д.)
        - event_config: настройки событий (начальные посты, горячие темы)
        - platform_configs: настройки платформ
        - generation_reasoning: пояснение LLM к генерации конфигурации
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)
        
        if not config:
            return jsonify({
                "success": False,
                "error": f"Конфигурация симуляции не найдена, сначала вызовите интерфейс /prepare"
            }), 404
        
        return jsonify({
            "success": True,
            "data": config
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения конфигурации: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/download', methods=['GET'])
def download_simulation_config(simulation_id: str):
    """Скачивание файла конфигурации симуляции"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return jsonify({
                "success": False,
                "error": "Файл конфигурации не существует, сначала вызовите интерфейс /prepare"
            }), 404
        
        return send_file(
            config_path,
            as_attachment=True,
            download_name="simulation_config.json"
        )
        
    except Exception as e:
        logger.error(f"Ошибка скачивания конфигурации: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/script/<script_name>/download', methods=['GET'])
def download_simulation_script(script_name: str):
    """
    Скачивание скриптов запуска симуляции (общие скрипты в backend/scripts/)
    
    Допустимые значения script_name:
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # Скрипты находятся в директории backend/scripts/
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        # Валидация имени скрипта
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py", 
            "run_parallel_simulation.py",
            "action_logger.py"
        ]
        
        if script_name not in allowed_scripts:
            return jsonify({
                "success": False,
                "error": f"Неизвестный скрипт: {script_name}, доступные: {allowed_scripts}"
            }), 400
        
        script_path = os.path.join(scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            return jsonify({
                "success": False,
                "error": f"Файл скрипта не существует: {script_name}"
            }), 404
        
        return send_file(
            script_path,
            as_attachment=True,
            download_name=script_name
        )
        
    except Exception as e:
        logger.error(f"Ошибка скачивания скрипта: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Интерфейс генерации Profile (автономное использование) ==============

@simulation_bp.route('/generate-profiles', methods=['POST'])
def generate_profiles():
    """
    Генерация OASIS Agent Profile напрямую из графа (без создания симуляции)
    
    Запрос (JSON):
        {
            "graph_id": "mirofish_xxxx",     // обязательно
            "entity_types": ["Student"],      // необязательно
            "use_llm": true,                  // необязательно
            "platform": "reddit"              // необязательно
        }
    """
    try:
        data = request.get_json() or {}
        
        graph_id = data.get('graph_id')
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Укажите graph_id"
            }), 400
        
        entity_types = data.get('entity_types')
        use_llm = data.get('use_llm', True)
        platform = data.get('platform', 'reddit')
        
        reader = ZepEntityReader()
        filtered = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=True
        )
        
        if filtered.filtered_count == 0:
            return jsonify({
                "success": False,
                "error": "Не найдено сущностей, соответствующих условиям"
            }), 400
        
        generator = OasisProfileGenerator()
        profiles = generator.generate_profiles_from_entities(
            entities=filtered.entities,
            use_llm=use_llm
        )
        
        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "entity_types": list(filtered.entity_types),
                "count": len(profiles_data),
                "profiles": profiles_data
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации Profile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Интерфейс управления запуском симуляции ==============

@simulation_bp.route('/start', methods=['POST'])
def start_simulation():
    """
    Запуск симуляции

    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",          // обязательно, ID симуляции
            "platform": "parallel",                // необязательно: twitter / reddit / parallel (по умолчанию)
            "max_rounds": 100,                     // необязательно: максимальное кол-во раундов, для ограничения длительных симуляций
            "enable_graph_memory_update": false,   // необязательно: обновлять ли активность Agent в памяти графа Zep в реальном времени
            "force": false                         // необязательно: принудительный перезапуск (остановит текущую симуляцию и очистит логи)
        }

    О параметре force:
        - При включении, если симуляция запущена или завершена — сначала останавливается и очищаются логи
        - Очищаются: run_state.json, actions.jsonl, simulation.log и др.
        - Конфигурационные файлы (simulation_config.json) и profile не удаляются
        - Используется при необходимости перезапуска симуляции

    О параметре enable_graph_memory_update:
        - При включении все действия Agent (посты, комментарии, лайки и т.д.) обновляются в графе Zep в реальном времени
        - Это позволяет графу "запоминать" процесс симуляции для последующего анализа или AI-диалога
        - Требуется валидный graph_id у связанного проекта
        - Используется пакетный механизм обновления для уменьшения количества API-вызовов

    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "process_pid": 12345,
                "twitter_running": true,
                "reddit_running": true,
                "started_at": "2025-12-01T10:00:00",
                "graph_memory_update_enabled": true,  // включено ли обновление памяти графа
                "force_restarted": true               // был ли это принудительный перезапуск
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400

        platform = data.get('platform', 'parallel')
        max_rounds = data.get('max_rounds')  # Необязательно: максимальное кол-во раундов
        enable_graph_memory_update = data.get('enable_graph_memory_update', False)  # Необязательно: обновление памяти графа
        force = data.get('force', False)  # Необязательно: принудительный перезапуск

        # Валидация параметра max_rounds
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify({
                        "success": False,
                        "error": "max_rounds должен быть положительным целым числом"
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": "max_rounds должен быть валидным целым числом"
                }), 400

        if platform not in ['twitter', 'reddit', 'parallel']:
            return jsonify({
                "success": False,
                "error": f"Недопустимый тип платформы: {platform}, доступные: twitter/reddit/parallel"
            }), 400

        # Проверка готовности симуляции
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({
                "success": False,
                "error": f"Симуляция не найдена: {simulation_id}"
            }), 404

        force_restarted = False
        
        # Интеллектуальная обработка статуса: если подготовка завершена — разрешаем перезапуск
        if state.status != SimulationStatus.READY:
            # Проверка завершённости подготовки
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)

            if is_prepared:
                # Подготовка завершена, проверяем наличие запущенных процессов
                if state.status == SimulationStatus.RUNNING:
                    # Проверка фактической работы процесса симуляции
                    run_state = SimulationRunner.get_run_state(simulation_id)
                    if run_state and run_state.runner_status.value == "running":
                        # Процесс действительно запущен
                        if force:
                            # Принудительный режим: остановка запущенной симуляции
                            logger.info(f"Принудительный режим: остановка запущенной симуляции {simulation_id}")
                            try:
                                SimulationRunner.stop_simulation(simulation_id)
                            except Exception as e:
                                logger.warning(f"Предупреждение при остановке симуляции: {str(e)}")
                        else:
                            return jsonify({
                                "success": False,
                                "error": f"Симуляция запущена, сначала вызовите /stop для остановки или используйте force=true для принудительного перезапуска"
                            }), 400

                # Принудительный режим — очистка логов выполнения
                if force:
                    logger.info(f"Принудительный режим: очистка логов симуляции {simulation_id}")
                    cleanup_result = SimulationRunner.cleanup_simulation_logs(simulation_id)
                    if not cleanup_result.get("success"):
                        logger.warning(f"Предупреждение при очистке логов: {cleanup_result.get('errors')}")
                    force_restarted = True

                # Процесс не существует или завершён, сброс статуса на ready
                logger.info(f"Симуляция {simulation_id} подготовка завершена, сброс статуса на ready (предыдущий статус: {state.status.value})")
                state.status = SimulationStatus.READY
                manager._save_simulation_state(state)
            else:
                # Подготовка не завершена
                return jsonify({
                    "success": False,
                    "error": f"Симуляция не готова, текущий статус: {state.status.value}, сначала вызовите интерфейс /prepare"
                }), 400
        
        # Получение graph_id (для обновления памяти графа)
        graph_id = None
        if enable_graph_memory_update:
            # Получение graph_id из состояния симуляции или проекта
            graph_id = state.graph_id
            if not graph_id:
                # Попытка получения из проекта
                project = ProjectManager.get_project(state.project_id)
                if project:
                    graph_id = project.graph_id
            
            if not graph_id:
                return jsonify({
                    "success": False,
                    "error": "Для обновления памяти графа требуется валидный graph_id, убедитесь что граф проекта построен"
                }), 400
            
            logger.info(f"Включено обновление памяти графа: simulation_id={simulation_id}, graph_id={graph_id}")
        
        # Запуск симуляции
        run_state = SimulationRunner.start_simulation(
            simulation_id=simulation_id,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id
        )
        
        # Обновление статуса симуляции
        state.status = SimulationStatus.RUNNING
        manager._save_simulation_state(state)
        
        response_data = run_state.to_dict()
        if max_rounds:
            response_data['max_rounds_applied'] = max_rounds
        response_data['graph_memory_update_enabled'] = enable_graph_memory_update
        response_data['force_restarted'] = force_restarted
        if enable_graph_memory_update:
            response_data['graph_id'] = graph_id
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Ошибка запуска симуляции: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/stop', methods=['POST'])
def stop_simulation():
    """
    Остановка симуляции
    
    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx"  // обязательно, ID симуляции
        }
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "stopped",
                "completed_at": "2025-12-01T12:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400
        
        run_state = SimulationRunner.stop_simulation(simulation_id)
        
        # Обновление статуса симуляции
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.PAUSED
            manager._save_simulation_state(state)
        
        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Ошибка остановки симуляции: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Интерфейс мониторинга статуса в реальном времени ==============

@simulation_bp.route('/<simulation_id>/run-status', methods=['GET'])
def get_run_status(simulation_id: str):
    """
    Получение статуса выполнения симуляции в реальном времени (для polling фронтенда)
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                "total_rounds": 144,
                "progress_percent": 3.5,
                "simulated_hours": 2,
                "total_simulation_hours": 72,
                "twitter_running": true,
                "reddit_running": true,
                "twitter_actions_count": 150,
                "reddit_actions_count": 200,
                "total_actions_count": 350,
                "started_at": "2025-12-01T10:00:00",
                "updated_at": "2025-12-01T10:30:00"
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        
        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "current_round": 0,
                    "total_rounds": 0,
                    "progress_percent": 0,
                    "twitter_actions_count": 0,
                    "reddit_actions_count": 0,
                    "total_actions_count": 0,
                }
            })
        
        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения статуса выполнения: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/run-status/detail', methods=['GET'])
def get_run_status_detail(simulation_id: str):
    """
    Получение детального статуса выполнения симуляции (со всеми действиями)
    
    Для отображения динамики в реальном времени на фронтенде
    
    Query-параметры:
        platform: Фильтр платформы (twitter/reddit, необязательно)
    
    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                ...
                "all_actions": [
                    {
                        "round_num": 5,
                        "timestamp": "2025-12-01T10:30:00",
                        "platform": "twitter",
                        "agent_id": 3,
                        "agent_name": "Agent Name",
                        "action_type": "CREATE_POST",
                        "action_args": {"content": "..."},
                        "result": null,
                        "success": true
                    },
                    ...
                ],
                "twitter_actions": [...],  # Все действия платформы Twitter
                "reddit_actions": [...]    # Все действия платформы Reddit
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        platform_filter = request.args.get('platform')
        
        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "all_actions": [],
                    "twitter_actions": [],
                    "reddit_actions": []
                }
            })
        
        # Получение полного списка действий
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter
        )
        
        # Получение действий по платформам
        twitter_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="twitter"
        ) if not platform_filter or platform_filter == "twitter" else []
        
        reddit_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="reddit"
        ) if not platform_filter or platform_filter == "reddit" else []
        
        # Получение действий текущего раунда (recent_actions — только последний раунд)
        current_round = run_state.current_round
        recent_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter,
            round_num=current_round
        ) if current_round > 0 else []
        
        # Получение базовой информации о статусе
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions показывает только содержимое обеих платформ за последний раунд
        result["recent_actions"] = [a.to_dict() for a in recent_actions]
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения детального статуса: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/actions', methods=['GET'])
def get_simulation_actions(simulation_id: str):
    """
    Получение истории действий Agent в симуляции
    
    Query-параметры:
        limit: Кол-во результатов (по умолчанию 100)
        offset: Смещение (по умолчанию 0)
        platform: Фильтр платформы (twitter/reddit)
        agent_id: Фильтр по Agent ID
        round_num: Фильтр по раунду
    
    Ответ:
        {
            "success": true,
            "data": {
                "count": 100,
                "actions": [...]
            }
        }
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        platform = request.args.get('platform')
        agent_id = request.args.get('agent_id', type=int)
        round_num = request.args.get('round_num', type=int)
        
        actions = SimulationRunner.get_actions(
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(actions),
                "actions": [a.to_dict() for a in actions]
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения истории действий: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/timeline', methods=['GET'])
def get_simulation_timeline(simulation_id: str):
    """
    Получение временной шкалы симуляции (сводка по раундам)
    
    Для отображения прогресс-бара и таймлайна на фронтенде
    
    Query-параметры:
        start_round: Начальный раунд (по умолчанию 0)
        end_round: Конечный раунд (по умолчанию все)
    
    Возвращает сводную информацию по каждому раунду
    """
    try:
        start_round = request.args.get('start_round', 0, type=int)
        end_round = request.args.get('end_round', type=int)
        
        timeline = SimulationRunner.get_timeline(
            simulation_id=simulation_id,
            start_round=start_round,
            end_round=end_round
        )
        
        return jsonify({
            "success": True,
            "data": {
                "rounds_count": len(timeline),
                "timeline": timeline
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения временной шкалы: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/agent-stats', methods=['GET'])
def get_agent_stats(simulation_id: str):
    """
    Получение статистики каждого Agent
    
    Для отображения рейтинга активности Agent, распределения действий и пр.
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id)
        
        return jsonify({
            "success": True,
            "data": {
                "agents_count": len(stats),
                "stats": stats
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики Agent: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Интерфейс запросов к базе данных ==============

@simulation_bp.route('/<simulation_id>/posts', methods=['GET'])
def get_simulation_posts(simulation_id: str):
    """
    Получение постов симуляции
    
    Query-параметры:
        platform: Тип платформы (twitter/reddit)
        limit: Кол-во результатов (по умолчанию 50)
        offset: Смещение
    
    Возвращает список постов (чтение из базы данных SQLite)
    """
    try:
        platform = request.args.get('platform', 'reddit')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )
        
        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)
        
        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "platform": platform,
                    "count": 0,
                    "posts": [],
                    "message": "База данных не существует, симуляция возможно ещё не запускалась"
                }
            })
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM post 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            posts = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute("SELECT COUNT(*) FROM post")
            total = cursor.fetchone()[0]
            
        except sqlite3.OperationalError:
            posts = []
            total = 0
        
        conn.close()
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "total": total,
                "count": len(posts),
                "posts": posts
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения постов: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/comments', methods=['GET'])
def get_simulation_comments(simulation_id: str):
    """
    Получение комментариев симуляции (только Reddit)
    
    Query-параметры:
        post_id: Фильтр по ID поста (необязательно)
        limit: Кол-во результатов
        offset: Смещение
    """
    try:
        post_id = request.args.get('post_id')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )
        
        db_path = os.path.join(sim_dir, "reddit_simulation.db")
        
        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "count": 0,
                    "comments": []
                }
            })
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if post_id:
                cursor.execute("""
                    SELECT * FROM comment 
                    WHERE post_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (post_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM comment 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            comments = [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.OperationalError:
            comments = []
        
        conn.close()
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(comments),
                "comments": comments
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения комментариев: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Интерфейс интервью Interview ==============

@simulation_bp.route('/interview', methods=['POST'])
def interview_agent():
    """
    Интервью с одним Agent

    Примечание: для этой функции требуется запущенная среда симуляции (после завершения цикла симуляции в режиме ожидания команд)

    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",       // обязательно, ID симуляции
            "agent_id": 0,                     // обязательно, Agent ID
            "prompt": "Что вы думаете об этом?",  // обязательно, вопрос интервью
            "platform": "twitter",             // необязательно, указание платформы (twitter/reddit)
                                               // если не указано: интервью на обеих платформах одновременно
            "timeout": 60                      // необязательно, таймаут (секунды), по умолчанию 60
        }

    Ответ (без указания platform, двухплатформенный режим):
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "Что вы думаете об этом?",
                "result": {
                    "agent_id": 0,
                    "prompt": "...",
                    "platforms": {
                        "twitter": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit": {"agent_id": 0, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }

    Ответ (с указанием platform):
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "Что вы думаете об этом?",
                "result": {
                    "agent_id": 0,
                    "response": "Я считаю...",
                    "platform": "twitter",
                    "timestamp": "2025-12-08T10:00:00"
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        agent_id = data.get('agent_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Необязательно: twitter/reddit/None
        timeout = data.get('timeout', 60)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400
        
        if agent_id is None:
            return jsonify({
                "success": False,
                "error": "Укажите agent_id"
            }), 400
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "Укажите prompt (вопрос интервью)"
            }), 400
        
        # Валидация параметра platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "Параметр platform может быть только 'twitter' или 'reddit'"
            }), 400
        
        # Проверка состояния среды
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "Среда симуляции не запущена или закрыта. Убедитесь что симуляция завершена и находится в режиме ожидания команд."
            }), 400
        
        # Оптимизация prompt, добавление префикса для предотвращения вызова инструментов Agent
        optimized_prompt = optimize_interview_prompt(prompt)
        
        result = SimulationRunner.interview_agent(
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Таймаут ожидания ответа Interview: {str(e)}"
        }), 504
        
    except Exception as e:
        logger.error(f"Ошибка Interview: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/batch', methods=['POST'])
def interview_agents_batch():
    """
    Пакетное интервью с несколькими Agent

    Примечание: для этой функции требуется запущенная среда симуляции

    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",       // обязательно, ID симуляции
            "interviews": [                    // обязательно, список интервью
                {
                    "agent_id": 0,
                    "prompt": "Что вы думаете об A?",
                    "platform": "twitter"      // необязательно, платформа интервью для данного Agent
                },
                {
                    "agent_id": 1,
                    "prompt": "Что вы думаете об B?"  // если platform не указан — используется значение по умолчанию
                }
            ],
            "platform": "reddit",              // необязательно, платформа по умолчанию (перекрывается platform каждого элемента)
                                               // если не указано: интервью каждого Agent на обеих платформах
            "timeout": 120                     // необязательно, таймаут (секунды), по умолчанию 120
        }

    Ответ:
        {
            "success": true,
            "data": {
                "interviews_count": 2,
                "result": {
                    "interviews_count": 4,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        "twitter_1": {"agent_id": 1, "response": "...", "platform": "twitter"},
                        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        interviews = data.get('interviews')
        platform = data.get('platform')  # Необязательно: twitter/reddit/None
        timeout = data.get('timeout', 120)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify({
                "success": False,
                "error": "Укажите interviews (список интервью)"
            }), 400

        # Валидация параметра platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "Параметр platform может быть только 'twitter' или 'reddit'"
            }), 400

        # Валидация каждого элемента интервью
        for i, interview in enumerate(interviews):
            if 'agent_id' not in interview:
                return jsonify({
                    "success": False,
                    "error": f"В элементе {i+1} списка интервью отсутствует agent_id"
                }), 400
            if 'prompt' not in interview:
                return jsonify({
                    "success": False,
                    "error": f"В элементе {i+1} списка интервью отсутствует prompt"
                }), 400
            # Валидация platform каждого элемента (если указан)
            item_platform = interview.get('platform')
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify({
                    "success": False,
                    "error": f"Параметр platform элемента {i+1} может быть только 'twitter' или 'reddit'"
                }), 400

        # Проверка состояния среды
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "Среда симуляции не запущена или закрыта. Убедитесь что симуляция завершена и находится в режиме ожидания команд."
            }), 400

        # Оптимизация prompt каждого элемента, добавление префикса для предотвращения вызова инструментов Agent
        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview['prompt'] = optimize_interview_prompt(interview.get('prompt', ''))
            optimized_interviews.append(optimized_interview)

        result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Таймаут ожидания пакетного Interview: {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Ошибка пакетного Interview: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/all', methods=['POST'])
def interview_all_agents():
    """
    Глобальное интервью — один вопрос всем Agent

    Примечание: для этой функции требуется запущенная среда симуляции

    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",            // обязательно, ID симуляции
            "prompt": "Что вы думаете об этом в целом?",  // обязательно, вопрос (одинаковый для всех Agent)
            "platform": "reddit",                   // необязательно, указание платформы (twitter/reddit)
                                                    // если не указано: интервью каждого Agent на обеих платформах
            "timeout": 180                          // необязательно, таймаут (секунды), по умолчанию 180
        }

    Ответ:
        {
            "success": true,
            "data": {
                "interviews_count": 50,
                "result": {
                    "interviews_count": 100,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        ...
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Необязательно: twitter/reddit/None
        timeout = data.get('timeout', 180)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Укажите prompt (вопрос интервью)"
            }), 400

        # Валидация параметра platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "Параметр platform может быть только 'twitter' или 'reddit'"
            }), 400

        # Проверка состояния среды
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "Среда симуляции не запущена или закрыта. Убедитесь что симуляция завершена и находится в режиме ожидания команд."
            }), 400

        # Оптимизация prompt, добавление префикса для предотвращения вызова инструментов Agent
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_all_agents(
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Таймаут ожидания глобального Interview: {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Ошибка глобального Interview: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/history', methods=['POST'])
def get_interview_history():
    """
    Получение истории Interview

    Чтение всех записей Interview из базы данных симуляции

    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",  // обязательно, ID симуляции
            "platform": "reddit",          // необязательно, тип платформы (reddit/twitter)
                                           // если не указано — возвращается история обеих платформ
            "agent_id": 0,                 // необязательно, только история интервью данного Agent
            "limit": 100                   // необязательно, кол-во результатов, по умолчанию 100
        }

    Ответ:
        {
            "success": true,
            "data": {
                "count": 10,
                "history": [
                    {
                        "agent_id": 0,
                        "response": "Я считаю...",
                        "prompt": "Что вы думаете об этом?",
                        "timestamp": "2025-12-08T10:00:00",
                        "platform": "reddit"
                    },
                    ...
                ]
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        platform = data.get('platform')  # Без указания — возвращается история обеих платформ
        agent_id = data.get('agent_id')
        limit = data.get('limit', 100)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400

        history = SimulationRunner.get_interview_history(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit
        )

        return jsonify({
            "success": True,
            "data": {
                "count": len(history),
                "history": history
            }
        })

    except Exception as e:
        logger.error(f"Ошибка получения истории Interview: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/env-status', methods=['POST'])
def get_env_status():
    """
    Получение статуса среды симуляции

    Проверка активности среды симуляции (может ли принимать команды Interview)

    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx"  // обязательно, ID симуляции
        }

    Ответ:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "Среда запущена, может принимать команды Interview"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400

        env_alive = SimulationRunner.check_env_alive(simulation_id)
        
        # Получение более детальной информации о статусе
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = "Среда запущена, может принимать команды Interview"
        else:
            message = "Среда не запущена или закрыта"

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "env_alive": env_alive,
                "twitter_available": env_status.get("twitter_available", False),
                "reddit_available": env_status.get("reddit_available", False),
                "message": message
            }
        })

    except Exception as e:
        logger.error(f"Ошибка получения статуса среды: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/close-env', methods=['POST'])
def close_simulation_env():
    """
    Закрытие среды симуляции
    
    Отправка команды закрытия среды симуляции для корректного выхода из режима ожидания.
    
    Примечание: отличается от /stop — /stop принудительно завершает процесс,
    а этот интерфейс корректно закрывает среду и выходит.
    
    Запрос (JSON):
        {
            "simulation_id": "sim_xxxx",  // обязательно, ID симуляции
            "timeout": 30                  // необязательно, таймаут (секунды), по умолчанию 30
        }
    
    Ответ:
        {
            "success": true,
            "data": {
                "message": "Команда закрытия среды отправлена",
                "result": {...},
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        timeout = data.get('timeout', 30)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Укажите simulation_id"
            }), 400
        
        result = SimulationRunner.close_simulation_env(
            simulation_id=simulation_id,
            timeout=timeout
        )
        
        # Обновление статуса симуляции
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)
        
        return jsonify({
            "success": result.get("success", False),
            "data": result
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Ошибка закрытия среды: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
