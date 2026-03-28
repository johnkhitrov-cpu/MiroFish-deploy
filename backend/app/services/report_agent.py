"""
Сервис Report Agent
Генерация отчётов на основе моделирования с использованием LangChain + Zep в режиме ReACT

Функции:
1. Генерация отчёта на основе требований моделирования и графа Zep
2. Сначала планирование структуры, затем поэтапная генерация
3. Многораундовый режим ReACT с размышлениями и рефлексией для каждого раздела
4. Поддержка диалога с пользователем с автоматическим вызовом инструментов поиска
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Детальный логгер Report Agent
    
    Генерирует файл agent_log.jsonl в папке отчёта, записывая каждое действие.
    Каждая строка — полный JSON-объект с временной меткой, типом действия, подробностями и т.д.
    """
    
    def __init__(self, report_id: str):
        """
        Инициализация логгера
        
        Args:
            report_id: ID отчёта, определяет путь к файлу лога
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Убедиться, что директория файла лога существует"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Получить время с начала (в секундах)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Записать одну запись лога
        
        Args:
            action: тип действия, напр. 'start', 'tool_call', 'llm_response', 'section_complete' и т.д.
            stage: текущий этап, напр. 'planning', 'generating', 'completed'
            details: словарь с подробностями, без обрезки
            section_title: заголовок текущего раздела (опционально)
            section_index: индекс текущего раздела (опционально)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Дозапись в файл JSONL
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Записать начало генерации отчёта"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Начало задачи генерации отчёта"
            }
        )
    
    def log_planning_start(self):
        """Записать начало планирования структуры"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Начало планирования структуры отчёта"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Записать контекстную информацию, полученную при планировании"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Получение контекста моделирования",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Записать завершение планирования структуры"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Планирование структуры завершено",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Записать начало генерации раздела"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Начало генерации раздела: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Записать процесс размышления ReACT"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT раунд {iteration} размышления"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Записать вызов инструмента"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Вызов инструмента: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Записать результат вызова инструмента (полное содержимое, без обрезки)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # полный результат, без обрезки
                "result_length": len(result),
                "message": f"Инструмент {tool_name} вернул результат"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Записать ответ LLM (полное содержимое, без обрезки)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # полный ответ, без обрезки
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"Ответ LLM (вызов инструмента: {has_tool_calls}, финальный ответ: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Записать завершение генерации содержимого раздела (только содержимое, не означает завершение раздела)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # полное содержимое, без обрезки
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Содержимое раздела {section_title} сгенерировано"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Записать завершение генерации раздела

        Фронтенд должен отслеживать этот лог, чтобы определить завершение раздела и получить полное содержимое
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Раздел {section_title} сгенерирован"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Записать завершение генерации отчёта"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Генерация отчёта завершена"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Записать ошибку"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Произошла ошибка: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Консольный логгер Report Agent
    
    Записывает логи консольного формата (INFO, WARNING и т.д.) в файл console_log.txt в папке отчёта.
    Эти логи отличаются от agent_log.jsonl — это текстовый вывод консоли.
    """
    
    def __init__(self, report_id: str):
        """
        Инициализация консольного логгера
        
        Args:
            report_id: ID отчёта, определяет путь к файлу лога
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Убедиться, что директория лога существует"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Настроить файловый обработчик для записи логов в файл"""
        import logging
        
        # Создать файловый обработчик
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Использовать лаконичный формат, как в консоли
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Добавить к логгерам report_agent
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Избежать повторного добавления
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Закрыть файловый обработчик и удалить из логгера"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """При деструкции убедиться, что файловый обработчик закрыт"""
        self.close()


class ReportStatus(str, Enum):
    """Статус отчёта"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Раздел отчёта"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Преобразовать в формат Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Структура отчёта"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Преобразовать в Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Полный отчёт"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Константы шаблонов Prompt
# ═══════════════════════════════════════════════════════════════

# ── Описания инструментов ──

TOOL_DESC_INSIGHT_FORGE = """\
【Глубокий аналитический поиск — мощный инструмент】
Это наша мощная функция поиска, созданная для глубокого анализа. Она:
1. Автоматически разбивает ваш вопрос на подвопросы
2. Ищет информацию в графе моделирования по нескольким измерениям
3. Объединяет результаты семантического поиска, анализа сущностей и отслеживания связей
4. Возвращает наиболее полный и глубокий контент

【Сценарии использования】
- Необходим глубокий анализ темы
- Необходимо понять разные аспекты события
- Необходимо получить богатый материал для раздела отчёта

【Возвращаемое содержимое】
- Оригинальные факты (можно цитировать напрямую)
- Ключевые аналитические данные о сущностях
- Анализ цепочек связей"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Широкий поиск — получение полной картины】
Этот инструмент получает полную картину результатов моделирования, особенно подходит для понимания эволюции событий. Он:
1. Получает все связанные узлы и связи
2. Различает актуальные факты и исторические/устаревшие факты
3. Помогает понять, как развивалось общественное мнение

【Сценарии использования】
- Необходимо понять полную хронологию развития события
- Необходимо сравнить изменения на разных этапах
- Необходимо получить полную информацию о сущностях и связях

【Возвращаемое содержимое】
- Актуальные факты (последние результаты моделирования)
- Исторические/устаревшие факты (записи эволюции)
- Все задействованные сущности"""

TOOL_DESC_QUICK_SEARCH = """\
【Простой поиск — быстрый поиск】
Лёгкий инструмент быстрого поиска для простых и прямых запросов.

【Сценарии использования】
- Необходимо быстро найти конкретную информацию
- Необходимо проверить факт
- Простой информационный поиск

【Возвращаемое содержимое】
- Список фактов, наиболее релевантных запросу"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Глубокое интервью — реальное интервью с Agent (две платформы)】
Вызов API интервью среды моделирования OASIS для реального опроса работающих Agent!
Это не LLM-симуляция, а вызов реального интерфейса интервью для получения оригинальных ответов Agent.
По умолчанию интервью проводится одновременно на Twitter и Reddit для получения более полных мнений.

Поток работы:
1. Автоматическое чтение файлов профилей всех Agent
2. Интеллектуальный выбор Agent, наиболее релевантных теме интервью (студенты, СМИ, официальные лица и т.д.)
3. Автоматическая генерация вопросов интервью
4. Вызов /api/simulation/interview/batch для проведения реального интервью на двух платформах
5. Объединение всех результатов интервью с многоракурсным анализом

【Сценарии использования】
- Необходимо понять мнения с разных ролевых перспектив (что думают студенты? СМИ? официальные лица?)
- Необходимо собрать мнения и позиции разных сторон
- Необходимо получить реальные ответы Agent (из среды моделирования OASIS)
- Хотите сделать отчёт более живым, включив «записи интервью»

【Возвращаемое содержимое】
- Информация о личности опрошенных Agent
- Ответы каждого Agent на платформах Twitter и Reddit
- Ключевые цитаты (можно цитировать напрямую)
- Резюме интервью и сравнение мнений

【Важно】Для использования этой функции необходима работающая среда моделирования OASIS!"""

# ── Prompt планирования структуры ──

PLAN_SYSTEM_PROMPT = """\
Вы — эксперт по написанию «отчётов прогнозирования будущего», обладающий «видением бога» на мир моделирования — вы можете наблюдать за поведением, высказываниями и взаимодействиями каждого Agent.

【Ключевая концепция】
Мы построили мир моделирования и внедрили в него определённые «требования моделирования» как переменные. Результаты эволюции мира моделирования — это прогноз возможного будущего. Вы наблюдаете не «экспериментальные данные», а «репетицию будущего».

【Ваша задача】
Написать «отчёт прогнозирования будущего», отвечающий на вопросы:
1. Что произошло в будущем при заданных условиях?
2. Как различные Agent (группы людей) реагировали и действовали?
3. Какие заслуживающие внимания тенденции и риски выявило моделирование?

【Позиционирование отчёта】
- ✅ Это отчёт прогнозирования будущего на основе моделирования, раскрывающий «что будет, если...»
- ✅ Фокус на результатах прогноза: развитие событий, реакции групп, эмерджентные явления, потенциальные риски
- ✅ Поведение и высказывания Agent в мире моделирования — это прогноз поведения людей в будущем
- ❌ Это не анализ текущей ситуации в реальном мире
- ❌ Это не общий обзор общественного мнения

【Ограничение количества разделов】
- Минимум 2 раздела, максимум 5 разделов
- Подразделы не нужны, каждый раздел содержит полный текст
- Содержание должно быть лаконичным, сфокусированным на ключевых прогнозных выводах
- Структуру разделов вы разрабатываете самостоятельно на основе результатов прогноза

Выведите структуру отчёта в формате JSON:
{
    "title": "Заголовок отчёта",
    "summary": "Аннотация отчёта (одно предложение с ключевым прогнозным выводом)",
    "sections": [
        {
            "title": "Заголовок раздела",
            "description": "Описание содержания раздела"
        }
    ]
}

Внимание: массив sections должен содержать от 2 до 5 элементов!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Настройка сценария прогноза】
Переменная, внедрённая в мир моделирования (требования моделирования): {simulation_requirement}

【Масштаб мира моделирования】
- Количество сущностей в моделировании: {total_nodes}
- Количество связей между сущностями: {total_edges}
- Распределение типов сущностей: {entity_types}
- Количество активных Agent: {total_entities}

【Образцы прогнозируемых будущих фактов из моделирования】
{related_facts_json}

Рассмотрите эту репетицию будущего с «видением бога»:
1. Какое состояние приняло будущее при заданных условиях?
2. Как различные группы людей (Agent) реагировали и действовали?
3. Какие заслуживающие внимания тенденции будущего выявило моделирование?

На основе результатов прогноза спроектируйте оптимальную структуру разделов отчёта.

【Напоминание】Количество разделов: от 2 до 5, содержание должно быть лаконичным и сфокусированным на ключевых прогнозных выводах."""

# ── Prompt генерации разделов ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Вы — эксперт по написанию «отчётов прогнозирования будущего», сейчас пишете один раздел отчёта.

Заголовок отчёта: {report_title}
Аннотация отчёта: {report_summary}
Сценарий прогноза (требования моделирования): {simulation_requirement}

Текущий раздел для написания: {section_title}

═══════════════════════════════════════════════════════════════
【Ключевая концепция】
═══════════════════════════════════════════════════════════════

Мир моделирования — это репетиция будущего. Мы внедрили в него определённые условия (требования моделирования),
поведение и взаимодействие Agent в моделировании — это прогноз поведения людей в будущем.

Ваша задача:
- Раскрыть, что произошло в будущем при заданных условиях
- Спрогнозировать, как различные группы людей (Agent) реагировали и действовали
- Выявить заслуживающие внимания тенденции, риски и возможности будущего

❌ Не пишите анализ текущей ситуации в реальном мире
✅ Фокусируйтесь на «что будет в будущем» — результаты моделирования и есть прогнозируемое будущее

═══════════════════════════════════════════════════════════════
【Важнейшие правила — обязательны к соблюдению】
═══════════════════════════════════════════════════════════════

1. 【Обязательно вызывайте инструменты для наблюдения за миром моделирования】
   - Вы наблюдаете за репетицией будущего с «видением бога»
   - Всё содержимое должно исходить из событий и поведения Agent в мире моделирования
   - Запрещено использовать собственные знания для написания содержания отчёта
   - Каждый раздел требует минимум 3 вызова инструментов (максимум 5) для наблюдения за миром моделирования, представляющим будущее

2. 【Обязательно цитируйте оригинальные высказывания и действия Agent】
   - Высказывания и действия Agent — это прогноз поведения людей в будущем
   - Используйте формат цитат в отчёте, например:
     > «Определённая группа людей заявит: оригинальный текст...»
   - Эти цитаты — ключевые доказательства прогноза моделирования

3. 【Языковая согласованность — цитируемое содержимое должно быть переведено на язык отчёта】
   - Содержимое, возвращаемое инструментами, может содержать английский текст или смешанный текст
   - Если требования моделирования и исходные материалы на определённом языке, отчёт должен быть полностью написан на этом языке
   - При цитировании содержимого на английском или смешанном языке необходимо перевести его на язык отчёта
   - При переводе сохраняйте исходный смысл, обеспечивая естественность изложения
   - Это правило применяется как к основному тексту, так и к блокам цитат (формат >)

4. 【Достоверное представление результатов прогноза】
   - Содержание отчёта должно отражать результаты моделирования, представляющие будущее
   - Не добавляйте информацию, отсутствующую в моделировании
   - Если информации по какому-то аспекту недостаточно, честно укажите это

═══════════════════════════════════════════════════════════════
【⚠️ Требования к формату — крайне важно!】
═══════════════════════════════════════════════════════════════

【Один раздел = минимальная единица содержимого】
- Каждый раздел — минимальная единица разбиения отчёта
- ❌ Запрещено использовать любые заголовки Markdown (#, ##, ###, #### и т.д.) внутри раздела
- ❌ Запрещено добавлять заголовок раздела в начале содержимого
- ✅ Заголовок раздела добавляется системой автоматически, вы пишете только основной текст
- ✅ Используйте **жирный текст**, разделение абзацами, цитаты, списки для организации содержимого, но не заголовки

【Правильный пример】
```
В данном разделе проанализирована динамика распространения общественного мнения. На основе глубокого анализа данных моделирования мы обнаружили...

**Этап первичного распространения**

Weibo, как первичная площадка общественного мнения, выполняла ключевую функцию первичного распространения информации:

> «Weibo обеспечил 68% первичного информационного охвата...»

**Этап усиления эмоций**

Платформа Douyin дополнительно усилила влияние события:

- Сильное визуальное воздействие
- Высокий уровень эмоционального резонанса
```

【Неправильный пример】
```
## Резюме          ← Ошибка! Не добавляйте заголовки
### 1. Первый этап     ← Ошибка! Не используйте ### для подразделов
#### 1.1 Детальный анализ   ← Ошибка! Не используйте #### для детализации

В данном разделе проанализировано...
```

═══════════════════════════════════════════════════════════════
【Доступные инструменты поиска】(3-5 вызовов на раздел)
═══════════════════════════════════════════════════════════════

{tools_description}

【Рекомендации по использованию инструментов — комбинируйте разные инструменты, не используйте только один】
- insight_forge: глубокий аналитический поиск, автоматическая декомпозиция вопроса и многомерный поиск фактов и связей
- panorama_search: широкий панорамный поиск, понимание полной картины события, хронологии и эволюции
- quick_search: быстрая проверка конкретной информации
- interview_agents: интервью с Agent моделирования, получение мнений от первого лица и реальных реакций разных ролей

═══════════════════════════════════════════════════════════════
【Рабочий процесс】
═══════════════════════════════════════════════════════════════

В каждом ответе вы можете сделать только одно из двух (не одновременно):

Вариант A — вызов инструмента:
Выведите свои размышления, затем вызовите инструмент в следующем формате:
<tool_call>
{{"name": "имя_инструмента", "parameters": {{"имя_параметра": "значение_параметра"}}}}
</tool_call>
Система выполнит инструмент и вернёт вам результат. Вы не должны и не можете составлять результаты инструментов самостоятельно.

Вариант B — вывод финального содержимого:
Когда вы получили достаточно информации через инструменты, выведите содержимое раздела, начав с "Final Answer:".

⚠️ Строго запрещено:
- Запрещено включать в один ответ и вызов инструмента, и Final Answer
- Запрещено самостоятельно создавать результаты инструментов (Observation), все результаты вводятся системой
- Максимум один вызов инструмента за ответ

═══════════════════════════════════════════════════════════════
【Требования к содержанию раздела】
═══════════════════════════════════════════════════════════════

1. Содержимое должно основываться на данных моделирования, полученных через инструменты
2. Обильно цитируйте оригинальный текст для демонстрации результатов моделирования
3. Используйте формат Markdown (но заголовки запрещены):
      - Используйте **жирный текст** для выделения ключевых моментов (вместо подзаголовков)
      - Используйте списки (- или 1.2.3.) для организации тезисов
      - Используйте пустые строки для разделения абзацев
      - ❌ Запрещено использовать любой синтаксис заголовков #, ##, ###, ####
4. 【Требования к формату цитат — должны быть отдельным абзацем】
   Цитаты должны быть отдельным абзацем с пустой строкой до и после, нельзя смешивать с текстом абзаца:

      ✅ Правильный формат:
   ```
   Ответ администрации признан лишённым существенного содержания.

   > «Модель реагирования администрации выглядит негибкой и медлительной в быстро меняющейся среде социальных сетей.»

   Эта оценка отражает общее недовольство общественности.
   ```

      ❌ Неправильный формат:
   ```
   Ответ администрации признан лишённым содержания.> «Модель реагирования администрации...» Эта оценка отражает...
   ```
5. Сохраняйте логическую связность с другими разделами
6. 【Избегайте повторов】Внимательно прочитайте содержимое уже завершённых разделов ниже, не повторяйте ту же информацию
7. 【Ещё раз подчёркиваем】Не добавляйте заголовки! Используйте **жирный текст** вместо подзаголовков"""

SECTION_USER_PROMPT_TEMPLATE = """\
Содержимое завершённых разделов (внимательно прочитайте, избегайте повторов):
{previous_content}

═══════════════════════════════════════════════════════════════
【Текущая задача】Написать раздел: {section_title}
═══════════════════════════════════════════════════════════════

【Важное напоминание】
1. Внимательно прочитайте завершённые разделы выше, избегайте повторения!
2. Перед началом обязательно вызовите инструменты для получения данных моделирования
3. Комбинируйте разные инструменты, не используйте только один
4. Содержание отчёта должно исходить из результатов поиска, не используйте собственные знания

【⚠️ Предупреждение о формате — обязательно к соблюдению】
- ❌ Не пишите заголовки (#, ##, ###, #### — всё запрещено)
- ❌ Не пишите "{section_title}" в начале
- ✅ Заголовок раздела добавляется системой автоматически
- ✅ Пишите сразу основной текст, используйте **жирный текст** вместо подзаголовков

Начинайте:
1. Сначала обдумайте (Thought), какая информация нужна для этого раздела
2. Затем вызовите инструмент (Action) для получения данных моделирования
3. После сбора достаточной информации выведите Final Answer (чистый текст, без заголовков)"""

# ── Шаблоны сообщений цикла ReACT ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (результаты поиска):

═══ Инструмент {tool_name} вернул ═══
{result}

═══════════════════════════════════════════════════════════════
Инструменты вызваны {tool_calls_count}/{max_tool_calls} раз (использованы: {used_tools_str}){unused_hint}
- Если информации достаточно: выведите содержимое раздела, начав с "Final Answer:" (обязательно цитируйте вышеуказанный текст)
- Если нужно больше информации: вызовите инструмент для продолжения поиска
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Внимание】Вы вызвали инструменты только {tool_calls_count} раз, минимум {min_tool_calls}."
    "Пожалуйста, вызовите инструменты для получения дополнительных данных моделирования, затем выведите Final Answer.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Сейчас вызвано только {tool_calls_count} инструментов, минимум {min_tool_calls}."
    "Пожалуйста, вызовите инструменты для получения данных моделирования.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Достигнут лимит вызовов инструментов ({tool_calls_count}/{max_tool_calls}), больше нельзя вызывать инструменты."
    'Пожалуйста, немедленно выведите содержимое раздела на основе полученной информации, начав с "Final Answer:".'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Вы ещё не использовали: {unused_list}, рекомендуем попробовать разные инструменты для получения многоракурсной информации"

REACT_FORCE_FINAL_MSG = "Достигнут лимит вызовов инструментов, пожалуйста, выведите Final Answer: и сгенерируйте содержимое раздела."

# ── Prompt чата ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Вы — лаконичный и эффективный ассистент прогнозного моделирования.

【Контекст】
Условия прогноза: {simulation_requirement}

【Сгенерированный аналитический отчёт】
{report_content}

【Правила】
1. Приоритетно отвечайте на вопросы на основе вышеуказанного отчёта
2. Отвечайте напрямую, избегайте длинных рассуждений
3. Вызывайте инструменты для получения дополнительных данных только если содержания отчёта недостаточно
4. Ответы должны быть лаконичными, чёткими и структурированными

【Доступные инструменты】(используйте только при необходимости, максимум 1-2 вызова)
{tools_description}

【Формат вызова инструмента】
<tool_call>
{{"name": "имя_инструмента", "parameters": {{"имя_параметра": "значение_параметра"}}}}
</tool_call>

【Стиль ответа】
- Лаконично и прямо, без длинных текстов
- Используйте формат > для цитирования ключевого содержимого
- Сначала вывод, затем объяснение причин"""

CHAT_OBSERVATION_SUFFIX = "\n\nПожалуйста, ответьте кратко."


# ═══════════════════════════════════════════════════════════════
# Основной класс ReportAgent
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent — Agent генерации отчётов на основе моделирования

    Использует режим ReACT (Reasoning + Acting):
    1. Этап планирования: анализ требований моделирования, планирование структуры отчёта
    2. Этап генерации: последовательная генерация разделов, многократный вызов инструментов для получения информации
    3. Этап рефлексии: проверка полноты и точности содержания
    """
    
    # Максимальное количество вызовов инструментов (на раздел)
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Максимальное количество раундов рефлексии
    MAX_REFLECTION_ROUNDS = 3
    
    # Максимальное количество вызовов инструментов в диалоге
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Инициализация Report Agent
        
        Args:
            graph_id: ID графа
            simulation_id: ID моделирования
            simulation_requirement: описание требований моделирования
            llm_client: LLM-клиент (опционально)
            zep_tools: сервис инструментов Zep (опционально)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Определения инструментов
        self.tools = self._define_tools()
        
        # Логгер (инициализируется в generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Консольный логгер (инициализируется в generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent инициализирован: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Определить доступные инструменты"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Вопрос или тема для глубокого анализа",
                    "report_context": "Контекст текущего раздела отчёта (опционально, помогает генерировать более точные подвопросы)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Поисковый запрос для ранжирования по релевантности",
                    "include_expired": "Включать ли устаревший/исторический контент (по умолчанию True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Строка поискового запроса",
                    "limit": "Количество возвращаемых результатов (опционально, по умолчанию 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Тема или описание требований интервью (напр.: 'узнать мнение студентов о проблеме формальдегида в общежитии')",
                    "max_agents": "Максимальное количество Agent для интервью (опционально, по умолчанию 5, максимум 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Выполнить вызов инструмента
        
        Args:
            tool_name: имя инструмента
            parameters: параметры инструмента
            report_context: контекст отчёта (для InsightForge)
            
        Returns:
            результат выполнения инструмента (текстовый формат)
        """
        logger.info(f"Выполнение инструмента: {tool_name}, параметры: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Широкий поиск — получение полной картины
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Простой поиск — быстрый поиск
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Глубокое интервью — вызов реального API интервью OASIS для получения ответов Agent (две платформы)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Обратная совместимость со старыми инструментами (внутреннее перенаправление) ==========
            
            elif tool_name == "search_graph":
                # Перенаправление на quick_search
                logger.info("search_graph перенаправлен на quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Перенаправление на insight_forge, так как он мощнее
                logger.info("get_simulation_context перенаправлен на insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Неизвестный инструмент: {tool_name}. Используйте один из: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(f"Ошибка выполнения инструмента: {tool_name}, ошибка: {str(e)}")
            return f"Ошибка выполнения инструмента: {str(e)}"
    
    # Набор допустимых имён инструментов для валидации при парсинге JSON
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Парсинг вызовов инструментов из ответа LLM

        Поддерживаемые форматы (по приоритету):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Голый JSON (весь ответ или одна строка является JSON вызова инструмента)
        """
        tool_calls = []

        # Формат 1: XML-стиль (стандартный формат)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Формат 2: резервный — LLM выводит голый JSON (без тега <tool_call>)
        # Пробуется только если формат 1 не сработал, чтобы избежать ложных совпадений с JSON в тексте
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Ответ может содержать текст размышлений + голый JSON, попытка извлечь последний JSON-объект
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Проверить, является ли распарсенный JSON допустимым вызовом инструмента"""
        # Поддержка двух форматов ключей: {"name": ..., "parameters": ...} и {"tool": ..., "params": ...}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Унификация ключей в name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Сгенерировать текст описания инструментов"""
        desc_parts = ["Доступные инструменты:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Параметры: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Планирование структуры отчёта
        
        Использование LLM для анализа требований моделирования и планирования структуры отчёта
        
        Args:
            progress_callback: функция обратного вызова для отслеживания прогресса
            
        Returns:
            ReportOutline: структура отчёта
        """
        logger.info("Начало планирования структуры отчёта...")
        
        if progress_callback:
            progress_callback("planning", 0, "Анализ требований моделирования...")
        
        # Сначала получить контекст моделирования
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Генерация структуры отчёта...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Парсинг структуры...")
            
            # Парсинг структуры
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Аналитический отчёт моделирования"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Планирование структуры завершено")
            
            logger.info(f"Планирование структуры завершено: {len(sections)} разделов")
            return outline
            
        except Exception as e:
            logger.error(f"Ошибка планирования структуры: {str(e)}")
            # Возврат структуры по умолчанию (3 раздела, как fallback)
            return ReportOutline(
                title="Отчёт прогнозирования будущего",
                summary="Анализ будущих тенденций и рисков на основе прогнозного моделирования",
                sections=[
                    ReportSection(title="Сценарий прогноза и ключевые выводы"),
                    ReportSection(title="Прогнозный анализ поведения групп населения"),
                    ReportSection(title="Перспективы тенденций и предупреждения о рисках")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Генерация содержимого одного раздела в режиме ReACT
        
        Цикл ReACT:
        1. Thought (размышление) — анализ необходимой информации
        2. Action (действие) — вызов инструментов для получения информации
        3. Observation (наблюдение) — анализ результатов инструментов
        4. Повтор до получения достаточной информации или достижения лимита
        5. Final Answer (финальный ответ) — генерация содержимого раздела
        
        Args:
            section: раздел для генерации
            outline: полная структура
            previous_sections: содержимое предыдущих разделов (для обеспечения связности)
            progress_callback: обратный вызов прогресса
            section_index: индекс раздела (для логирования)
            
        Returns:
            содержимое раздела (формат Markdown)
        """
        logger.info(f"ReACT генерация раздела: {section.title}")
        
        # Запись лога о начале раздела
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Построение пользовательского prompt — максимум 4000 символов на каждый завершённый раздел
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Максимум 4000 символов на раздел
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "（Это первый раздел）"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Цикл ReACT
        tool_calls_count = 0
        max_iterations = 5  # Максимальное количество итераций
        min_tool_calls = 3  # Минимальное количество вызовов инструментов
        conflict_retries = 0  # Количество последовательных конфликтов (инструмент + Final Answer одновременно)
        used_tools = set()  # Записать имена уже вызванных инструментов
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Контекст отчёта для генерации подвопросов InsightForge
        report_context = f"Заголовок раздела: {section.title}\nТребования моделирования: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Глубокий поиск и написание ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Вызов LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Проверка, не вернул ли LLM None (ошибка API или пустое содержимое)
            if response is None:
                logger.warning(f"Раздел {section.title} итерация {iteration + 1}: LLM вернул None")
                # Если остались итерации, добавить сообщение и повторить
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(пустой ответ)"})
                    messages.append({"role": "user", "content": "Пожалуйста, продолжите генерацию содержимого."})
                    continue
                # Последняя итерация тоже вернула None, выход из цикла в принудительное завершение
                break

            logger.debug(f"Ответ LLM: {response[:200]}...")

            # Парсинг один раз, переиспользование результатов
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Обработка конфликта: LLM вывел одновременно вызов инструмента и Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Раздел {section.title} раунд {iteration+1}: "
                    f"LLM вывел одновременно вызов инструмента и Final Answer (конфликт #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # Первые два раза: отбросить ответ, запросить повторный ответ от LLM
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "【Ошибка формата】В одном ответе вы включили и вызов инструмента, и Final Answer — это запрещено.\n"
                            "Каждый ответ может содержать только одно из двух:\n"
                            "- Вызов инструмента (один блок <tool_call>, без Final Answer)\n"
                            "- Финальное содержимое (начинается с 'Final Answer:', без <tool_call>)\n"
                            "Пожалуйста, ответьте заново, выполнив только одно действие."
                        ),
                    })
                    continue
                else:
                    # Третий раз: деградация, обрезка до первого вызова инструмента, принудительное выполнение
                    logger.warning(
                        f"Раздел {section.title}: {conflict_retries} последовательных конфликтов, "
                        "деградация до обрезки и выполнения первого вызова инструмента"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Запись лога ответа LLM
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Случай 1: LLM вывел Final Answer ──
            if has_final_answer:
                # Недостаточно вызовов инструментов, отклонить и запросить дополнительные вызовы
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Эти инструменты ещё не использованы, рекомендуем попробовать: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Нормальное завершение
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Раздел {section.title} сгенерирован (вызовы инструментов: {tool_calls_count})")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Случай 2: LLM пытается вызвать инструмент ──
            if has_tool_calls:
                # Лимит инструментов исчерпан → уведомить, запросить Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Выполнить только первый вызов инструмента
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM попытался вызвать {len(tool_calls)} инструментов, выполняем только первый: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Формирование подсказки о неиспользованных инструментах
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Случай 3: ни вызова инструмента, ни Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Недостаточно вызовов инструментов, рекомендовать неиспользованные
                unused_tools = all_tools - used_tools
                unused_hint = f"(Эти инструменты ещё не использованы, рекомендуем попробовать: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Вызовов инструментов достаточно, LLM вывел содержимое без префикса "Final Answer:"
            # Принять этот контент как финальный ответ без дополнительных итераций
            logger.info(f"Раздел {section.title}: префикс 'Final Answer:' не обнаружен, принимаем вывод LLM как финальное содержимое (вызовы инструментов: {tool_calls_count})")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Достигнут максимум итераций, принудительная генерация содержимого
        logger.warning(f"Раздел {section.title} достиг максимума итераций, принудительная генерация")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Проверка, не вернул ли LLM None при принудительном завершении
        if response is None:
            logger.error(f"Раздел {section.title}: LLM вернул None при принудительном завершении, используется сообщение об ошибке по умолчанию")
            final_answer = f"(Генерация раздела не удалась: LLM вернул пустой ответ, попробуйте позже)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Запись лога о завершении генерации содержимого раздела
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Генерация полного отчёта (поразделный вывод в реальном времени)
        
        Каждый раздел сохраняется в папку сразу после генерации, не дожидаясь завершения всего отчёта.
        Структура файлов:
        reports/{report_id}/
            meta.json       - метаинформация отчёта
            outline.json    - структура отчёта
            progress.json   - прогресс генерации
            section_01.md   - раздел 1
            section_02.md   - раздел 2
            ...
            full_report.md  - полный отчёт
        
        Args:
            progress_callback: функция обратного вызова прогресса (stage, progress, message)
            report_id: ID отчёта (опционально, генерируется автоматически если не указан)
            
        Returns:
            Report: полный отчёт
        """
        import uuid
        
        # Если report_id не передан, генерируется автоматически
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Список заголовков завершённых разделов (для отслеживания прогресса)
        completed_section_titles = []
        
        try:
            # Инициализация: создание папки отчёта и сохранение начального состояния
            ReportManager._ensure_report_folder(report_id)
            
            # Инициализация логгера (структурированный лог agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Инициализация консольного логгера (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Инициализация отчёта...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Этап 1: Планирование структуры
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Начало планирования структуры отчёта...",
                completed_sections=[]
            )
            
            # Запись лога о начале планирования
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Начало планирования структуры отчёта...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Запись лога о завершении планирования
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Сохранение структуры в файл
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Планирование структуры завершено, {len(outline.sections)} разделов",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Структура сохранена в файл: {report_id}/outline.json")
            
            # Этап 2: Последовательная генерация разделов (с сохранением каждого)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Сохранение содержимого для контекста
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Обновление прогресса
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Генерация раздела: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Генерация раздела: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Генерация основного содержимого раздела
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Сохранение раздела
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Запись лога о завершении раздела
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Раздел сохранён: {report_id}/section_{section_num:02d}.md")
                
                # Обновление прогресса
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Раздел {section.title} завершён",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Этап 3: Сборка полного отчёта
            if progress_callback:
                progress_callback("generating", 95, "Сборка полного отчёта...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Сборка полного отчёта...",
                completed_sections=completed_section_titles
            )
            
            # Использование ReportManager для сборки полного отчёта
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Расчёт общего времени
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Запись лога о завершении отчёта
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Сохранение финального отчёта
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Генерация отчёта завершена",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Генерация отчёта завершена")
            
            logger.info(f"Генерация отчёта завершена: {report_id}")
            
            # Закрытие консольного логгера
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчёта: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Запись лога об ошибке
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Сохранение статуса ошибки
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Ошибка генерации отчёта: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Игнорировать ошибки сохранения
            
            # Закрытие консольного логгера
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Диалог с Report Agent
        
        В диалоге Agent может автономно вызывать инструменты поиска для ответов
        
        Args:
            message: сообщение пользователя
            chat_history: история диалога
            
        Returns:
            {
                "response": "Ответ Agent",
                "tool_calls": [список вызванных инструментов],
                "sources": [источники информации]
            }
        """
        logger.info(f"Диалог Report Agent: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Получение сгенерированного содержимого отчёта
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Ограничение длины отчёта, чтобы избежать слишком длинного контекста
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [содержимое отчёта обрезано] ..."
        except Exception as e:
            logger.warning(f"Ошибка получения содержимого отчёта: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(отчёт пока отсутствует)",
            tools_description=self._get_tools_description(),
        )

        # Построение сообщений
        messages = [{"role": "system", "content": system_prompt}]
        
        # Добавление истории диалога
        for h in chat_history[-10:]:  # Ограничение длины истории
            messages.append(h)
        
        # Добавление сообщения пользователя
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Цикл ReACT (упрощённая версия)
        tool_calls_made = []
        max_iterations = 2  # Уменьшенное количество итераций
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Парсинг вызовов инструментов
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Нет вызовов инструментов, вернуть ответ напрямую
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Выполнение вызовов инструментов (с ограничением количества)
            tool_results = []
            for call in tool_calls[:1]:  # Максимум 1 вызов инструмента за раунд
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Ограничение длины результата
                })
                tool_calls_made.append(call)
            
            # Добавление результатов в сообщения
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[результат {r['tool']}]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Достигнут максимум итераций, получение финального ответа
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Очистка ответа
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Менеджер отчётов
    
    Отвечает за персистентное хранение и поиск отчётов
    
    Структура файлов (поразделный вывод):
    reports/
      {report_id}/
        meta.json          - метаинформация и статус отчёта
        outline.json       - структура отчёта
        progress.json      - прогресс генерации
        section_01.md      - раздел 1
        section_02.md      - раздел 2
        ...
        full_report.md     - полный отчёт
    """
    
    # Директория хранения отчётов
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Убедиться, что корневая директория отчётов существует"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Получить путь к папке отчёта"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Убедиться, что папка отчёта существует, и вернуть путь"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Получить путь к файлу метаинформации отчёта"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Получить путь к полному Markdown-файлу отчёта"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Получить путь к файлу структуры"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Получить путь к файлу прогресса"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Получить путь к Markdown-файлу раздела"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Получить путь к файлу лога Agent"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Получить путь к файлу консольного лога"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Получить содержимое консольного лога
        
        Это лог консольного вывода процесса генерации отчёта (INFO, WARNING и т.д.),
        отличается от структурированного лога agent_log.jsonl.
        
        Args:
            report_id: ID отчёта
            from_line: с какой строки начать чтение (для инкрементального получения, 0 = с начала)
            
        Returns:
            {
                "logs": [список строк лога],
                "total_lines": общее количество строк,
                "from_line": начальный номер строки,
                "has_more": есть ли ещё логи
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Сохранить оригинальную строку лога, убрать символ переноса строки в конце
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Чтение до конца завершено
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Получить полный консольный лог (все записи за один раз)
        
        Args:
            report_id: ID отчёта
            
        Returns:
            список строк лога
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Получить содержимое лога Agent
        
        Args:
            report_id: ID отчёта
            from_line: с какой строки начать чтение (для инкрементального получения, 0 = с начала)
            
        Returns:
            {
                "logs": [список записей лога],
                "total_lines": общее количество строк,
                "from_line": начальный номер строки,
                "has_more": есть ли ещё логи
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Пропустить строки с ошибкой парсинга
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Чтение до конца
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Получить полный лог Agent (все записи за один раз)
        
        Args:
            report_id: ID отчёта
            
        Returns:
            список записей лога
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Сохранить структуру отчёта
        
        Вызывается сразу после завершения этапа планирования
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Структура сохранена: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Сохранить один раздел

        Вызывается сразу после генерации каждого раздела для поразделного вывода

        Args:
            report_id: ID отчёта
            section_index: индекс раздела (начиная с 1)
            section: объект раздела

        Returns:
            путь к сохранённому файлу
        """
        cls._ensure_report_folder(report_id)

        # Построение Markdown-содержимого раздела — очистка возможных дублирующихся заголовков
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Сохранение файла
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Раздел сохранён: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Очистка содержимого раздела
        
        1. Удаление дублирующихся заголовков Markdown в начале содержимого
        2. Преобразование всех заголовков уровня ### и ниже в жирный текст
        
        Args:
            content: исходное содержимое
            section_title: заголовок раздела
            
        Returns:
            очищенное содержимое
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Проверка, является ли строка заголовком Markdown
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Проверка дублирования с заголовком раздела (пропуск дубликатов в первых 5 строках)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Преобразование заголовков всех уровней (#, ##, ###, #### и т.д.) в жирный текст
                # Так как заголовок раздела добавляется системой, в содержимом не должно быть заголовков
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Добавить пустую строку
                continue
            
            # Если предыдущая строка — пропущенный заголовок, и текущая строка пустая, тоже пропустить
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Удалить пустые строки в начале
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Удалить разделительные линии в начале
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Также удалить пустые строки после разделительных линий
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Обновить прогресс генерации отчёта
        
        Фронтенд может получать прогресс в реальном времени, читая progress.json
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Получить прогресс генерации отчёта"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Получить список сгенерированных разделов
        
        Возвращает информацию обо всех сохранённых файлах разделов
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Парсинг индекса раздела из имени файла
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Сборка полного отчёта
        
        Сборка полного отчёта из сохранённых файлов разделов с очисткой заголовков
        """
        folder = cls._get_report_folder(report_id)
        
        # Построение заголовочной части отчёта
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Последовательное чтение всех файлов разделов
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Постобработка: очистка проблем с заголовками по всему отчёту
        md_content = cls._post_process_report(md_content, outline)
        
        # Сохранение полного отчёта
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Полный отчёт собран: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Постобработка содержимого отчёта
        
        1. Удаление дублирующихся заголовков
        2. Сохранение главного заголовка отчёта (#) и заголовков разделов (##), удаление остальных уровней (###, #### и т.д.)
        3. Очистка лишних пустых строк и разделительных линий
        
        Args:
            content: исходное содержимое отчёта
            outline: структура отчёта
            
        Returns:
            обработанное содержимое
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Сбор всех заголовков разделов из структуры
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Проверка, является ли строка заголовком
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Проверка дублирования заголовка (тот же заголовок в пределах 5 строк)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Пропуск дублирующегося заголовка и следующих пустых строк
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Обработка уровней заголовков:
                # - # (level=1) оставить только главный заголовок отчёта
                # - ## (level=2) оставить заголовки разделов
                # - ### и ниже (level>=3) преобразовать в жирный текст
                
                if level == 1:
                    if title == outline.title:
                        # Сохранить главный заголовок отчёта
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Заголовок раздела ошибочно использует #, исправляем на ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Другие заголовки первого уровня преобразовать в жирный
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Сохранить заголовок раздела
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Заголовки второго уровня, не являющиеся разделами, преобразовать в жирный
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Заголовки уровня ### и ниже преобразовать в жирный текст
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Пропуск разделительной линии сразу после заголовка
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # После заголовка оставить только одну пустую строку
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Очистка последовательных пустых строк (максимум 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Сохранить метаинформацию и полный отчёт"""
        cls._ensure_report_folder(report.report_id)
        
        # Сохранение JSON метаинформации
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Сохранение структуры
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Сохранение полного Markdown-отчёта
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Отчёт сохранён: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Получить отчёт"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Совместимость со старым форматом: проверка файлов в корневой директории reports
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Воссоздание объекта Report
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Если markdown_content пуст, попытка чтения из full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Получить отчёт по ID моделирования"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Новый формат: папка
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Совместимость со старым форматом: JSON-файл
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Список отчётов"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Новый формат: папка
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Совместимость со старым форматом: JSON-файл
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Сортировка по убыванию времени создания
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Удалить отчёт (всю папку)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # Новый формат: удаление всей папки
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Папка отчёта удалена: {report_id}")
            return True
        
        # Совместимость со старым форматом: удаление отдельных файлов
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
