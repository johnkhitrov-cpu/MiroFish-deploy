"""
Сервис инструментов поиска Zep
Инкапсулирует поиск по графу, чтение узлов, запросы рёбер и другие инструменты для Report Agent

Основные инструменты поиска (оптимизированные):
1. InsightForge (глубокий анализ) — мощный гибридный поиск, автогенерация подвопросов и многомерный поиск
2. PanoramaSearch (широкий поиск) — полная картина, включая устаревший контент
3. QuickSearch (быстрый поиск) — быстрый поиск
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """Результат поиска"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Преобразование в текстовый формат для понимания LLM"""
        text_parts = [f"Поисковый запрос: {self.query}", f"Найдено {self.total_count} релевантных записей"]
        
        if self.facts:
            text_parts.append("\n### Релевантные факты:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Информация об узле"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """Преобразование в текстовый формат"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Неизвестный тип")
        return f"Сущность: {self.name} (тип: {entity_type})\nРезюме: {self.summary}"


@dataclass
class EdgeInfo:
    """Информация о ребре"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Временная информация
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """Преобразование в текстовый формат"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Связь: {source} --[{self.name}]--> {target}\nФакт: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "Неизвестно"
            invalid_at = self.invalid_at or "По настоящее время"
            base_text += f"\nСрок действия: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Истёк: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """Истёк ли срок"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """Недействительно ли"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Результат глубокого анализа (InsightForge)
    Содержит результаты поиска по нескольким подвопросам и комплексный анализ
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # Результаты поиска по различным измерениям
    semantic_facts: List[str] = field(default_factory=list)  # Результаты семантического поиска
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Аналитика сущностей
    relationship_chains: List[str] = field(default_factory=list)  # Цепочки связей
    
    # Статистика
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """Преобразование в детальный текстовый формат для понимания LLM"""
        text_parts = [
            f"## Глубокий анализ прогноза будущего",
            f"Анализируемый вопрос: {self.query}",
            f"Сценарий прогноза: {self.simulation_requirement}",
            f"\n### Статистика данных прогноза",
            f"- Релевантных фактов прогноза: {self.total_facts}",
            f"- Задействовано сущностей: {self.total_entities}",
            f"- Цепочек связей: {self.total_relationships}"
        ]
        
        # Подвопросы
        if self.sub_queries:
            text_parts.append(f"\n### Анализируемые подвопросы")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # Результаты семантического поиска
        if self.semantic_facts:
            text_parts.append(f"\n### 【Ключевые факты】(цитируйте в отчёте дословно)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Аналитика сущностей
        if self.entity_insights:
            text_parts.append(f"\n### 【Ключевые сущности】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Неизвестно')}** ({entity.get('type', 'Сущность')})")
                if entity.get('summary'):
                    text_parts.append(f"  Резюме: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Связанных фактов: {len(entity.get('related_facts', []))}")
        
        # Цепочки связей
        if self.relationship_chains:
            text_parts.append(f"\n### 【Цепочки связей】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Результат широкого поиска (Panorama)
    Содержит всю связанную информацию, включая устаревший контент
    """
    query: str
    
    # Все узлы
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # Все рёбра (включая устаревшие)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Текущие действующие факты
    active_facts: List[str] = field(default_factory=list)
    # Устаревшие/недействительные факты (историческая запись)
    historical_facts: List[str] = field(default_factory=list)
    
    # Статистика
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """Преобразование в текстовый формат (полная версия, без обрезки)"""
        text_parts = [
            f"## Результат широкого поиска (панорамный вид будущего)",
            f"Запрос: {self.query}",
            f"\n### Статистика",
            f"- Всего узлов: {self.total_nodes}",
            f"- Всего рёбер: {self.total_edges}",
            f"- Текущих действующих фактов: {self.active_count}",
            f"- Исторических/устаревших фактов: {self.historical_count}"
        ]
        
        # Текущие действующие факты (полный вывод, без обрезки)
        if self.active_facts:
            text_parts.append(f"\n### 【Текущие действующие факты】(оригинальные результаты симуляции)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Исторические/устаревшие факты (полный вывод, без обрезки)
        if self.historical_facts:
            text_parts.append(f"\n### 【Исторические/устаревшие факты】(запись процесса эволюции)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Ключевые сущности (полный вывод, без обрезки)
        if self.all_nodes:
            text_parts.append(f"\n### 【Задействованные сущности】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Сущность")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Результат интервью с одним Agent"""
    agent_name: str
    agent_role: str  # Тип роли (напр.: студент, преподаватель, СМИ и т.д.)
    agent_bio: str  # Краткая биография
    question: str  # Вопрос интервью
    response: str  # Ответ на интервью
    key_quotes: List[str] = field(default_factory=list)  # Ключевые цитаты
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Отображение полного agent_bio, без обрезки
        text += f"_Биография: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Ключевые цитаты:**\n"
            for quote in self.key_quotes:
                # Очистка различных кавычек
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Удаление начальных знаков препинания
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Фильтрация мусорного контента с номерами вопросов (вопрос 1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Обрезка слишком длинного контента (по точке, а не жёсткая обрезка)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Результат интервью (Interview)
    Содержит ответы нескольких Agent симуляции
    """
    interview_topic: str  # Тема интервью
    interview_questions: List[str]  # Список вопросов интервью
    
    # Agent, выбранные для интервью
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Ответы каждого Agent на интервью
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # Обоснование выбора Agent
    selection_reasoning: str = ""
    # Сводное резюме интервью
    summary: str = ""
    
    # Статистика
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """Преобразование в детальный текстовый формат для понимания LLM и цитирования в отчёте"""
        text_parts = [
            "## Отчёт глубинного интервью",
            f"**Тема интервью:** {self.interview_topic}",
            f"**Кол-во интервьюируемых:** {self.interviewed_count} / {self.total_agents} Agent симуляции",
            "\n### Обоснование выбора интервьюируемых",
            self.selection_reasoning or "(Автоматический выбор)",
            "\n---",
            "\n### Протокол интервью",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Интервью #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(Нет записей интервью)\n\n---")

        text_parts.append("\n### Резюме интервью и ключевые тезисы")
        text_parts.append(self.summary or "(Нет резюме)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Сервис инструментов поиска Zep
    
    【Основные инструменты поиска — оптимизированные】
    1. insight_forge — глубокий анализ (самый мощный, автогенерация подвопросов, многомерный поиск)
    2. panorama_search — широкий поиск (полная картина, включая устаревший контент)
    3. quick_search — простой поиск (быстрый поиск)
    4. interview_agents — глубинное интервью (интервью с Agent симуляции, получение многоракурсных мнений)
    
    【Базовые инструменты】
    - search_graph — семантический поиск по графу
    - get_all_nodes — получение всех узлов графа
    - get_all_edges — получение всех рёбер графа (с временной информацией)
    - get_node_detail — получение детальной информации об узле
    - get_node_edges — получение рёбер, связанных с узлом
    - get_entities_by_type — получение сущностей по типу
    - get_entity_summary — получение резюме связей сущности
    """
    
    # Конфигурация повторных попыток
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY не настроен")
        
        self.client = Zep(api_key=self.api_key)
        # LLM-клиент для генерации подвопросов InsightForge
        self._llm_client = llm_client
        logger.info("ZepToolsService инициализация завершена")
    
    @property
    def llm(self) -> LLMClient:
        """Отложенная инициализация LLM-клиента"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """API-вызов с механизмом повторных попыток"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} попытка {attempt + 1} неудачна: {str(e)[:100]}, "
                        f"повтор через {delay:.1f} сек..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} не удалось после {max_retries} попыток: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Семантический поиск по графу
        
        Гибридный поиск (семантический + BM25) по графу.
        При недоступности Zep Cloud search API — откат на локальное сопоставление по ключевым словам.
        
        Args:
            graph_id: ID графа (Standalone Graph)
            query: Поисковый запрос
            limit: Количество результатов
            scope: Область поиска, "edges" или "nodes"
            
        Returns:
            SearchResult: Результат поиска
        """
        logger.info(f"Поиск по графу: graph_id={graph_id}, query={query[:50]}...")
        
        # Попытка использования Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"Поиск по графу(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Парсинг результатов поиска рёбер
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # Парсинг результатов поиска узлов
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # Резюме узла тоже считается фактом
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"Поиск завершён: найдено {len(facts)} релевантных фактов")
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep Search API неудачен, откат на локальный поиск: {str(e)}")
            # Откат: локальный поиск по ключевым словам
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Локальный поиск по ключевым словам (откат при недоступности Zep Search API)
        
        Получает все рёбра/узлы и выполняет локальное сопоставление по ключевым словам
        
        Args:
            graph_id: ID графа
            query: Поисковый запрос
            limit: Количество результатов
            scope: Область поиска
            
        Returns:
            SearchResult: Результат поиска
        """
        logger.info(f"Локальный поиск: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Извлечение ключевых слов запроса (простая токенизация)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """Вычисление оценки совпадения текста с запросом"""
            if not text:
                return 0
            text_lower = text.lower()
            # Полное совпадение с запросом
            if query_lower in text_lower:
                return 100
            # Совпадение ключевых слов
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Получение всех рёбер и сопоставление
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # Сортировка по оценке
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Получение всех узлов и сопоставление
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"Локальный поиск завершён: найдено {len(facts)} релевантных фактов")
            
        except Exception as e:
            logger.error(f"Ошибка локального поиска: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Получение всех узлов графа (с пагинацией)

        Args:
            graph_id: ID графа

        Returns:
            Список узлов
        """
        logger.info(f"Получение всех узлов графа {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"Получено {len(result)} узлов")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Получение всех рёбер графа (с пагинацией, включая временную информацию)

        Args:
            graph_id: ID графа
            include_temporal: Включать ли временную информацию (по умолчанию True)

        Returns:
            Список рёбер (включая created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(f"Получение всех рёбер графа {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # Добавление временной информации
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"Получено {len(result)} рёбер")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Получение детальной информации об узле
        
        Args:
            node_uuid: UUID узла
            
        Returns:
            Информация об узле или None
        """
        logger.info(f"Получение деталей узла: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"Получение деталей узла(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"Ошибка получения деталей узла: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Получение всех рёбер, связанных с узлом
        
        Получает все рёбра графа и фильтрует те, что связаны с указанным узлом
        
        Args:
            graph_id: ID графа
            node_uuid: UUID узла
            
        Returns:
            Список рёбер
        """
        logger.info(f"Получение рёбер узла {node_uuid[:8]}...")
        
        try:
            # Получение всех рёбер графа и фильтрация
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Проверка связи ребра с узлом (как источник или цель)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"Найдено {len(result)} рёбер, связанных с узлом")
            return result
            
        except Exception as e:
            logger.warning(f"Ошибка получения рёбер узла: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Получение сущностей по типу
        
        Args:
            graph_id: ID графа
            entity_type: Тип сущности (напр. Student, PublicFigure и т.д.)
            
        Returns:
            Список сущностей указанного типа
        """
        logger.info(f"Получение сущностей типа {entity_type}...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Проверка наличия указанного типа в labels
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"Найдено {len(filtered)} сущностей типа {entity_type}")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Получение резюме связей сущности
        
        Поиск всей информации, связанной с сущностью, и генерация резюме
        
        Args:
            graph_id: ID графа
            entity_name: Имя сущности
            
        Returns:
            Информация резюме сущности
        """
        logger.info(f"Получение резюме связей сущности {entity_name}...")
        
        # Сначала поиск информации, связанной с сущностью
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # Попытка найти сущность среди всех узлов
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # Передача параметра graph_id
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Получение статистики графа
        
        Args:
            graph_id: ID графа
            
        Returns:
            Статистическая информация
        """
        logger.info(f"Получение статистики графа {graph_id}...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # Статистика распределения типов сущностей
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # Статистика распределения типов связей
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Получение контекстной информации для симуляции
        
        Комплексный поиск всей информации, связанной с требованиями симуляции
        
        Args:
            graph_id: ID графа
            simulation_requirement: Описание требований симуляции
            limit: Лимит количества для каждого типа информации
            
        Returns:
            Контекстная информация симуляции
        """
        logger.info(f"Получение контекста симуляции: {simulation_requirement[:50]}...")
        
        # Поиск информации, связанной с требованиями симуляции
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Получение статистики графа
        stats = self.get_graph_statistics(graph_id)
        
        # Получение всех узлов сущностей
        all_nodes = self.get_all_nodes(graph_id)
        
        # Фильтрация сущностей с реальным типом (не чисто Entity узлы)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Ограничение количества
            "total_entities": len(entities)
        }
    
    # ========== Основные инструменты поиска (оптимизированные) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        【InsightForge — глубокий анализ】
        
        Самая мощная гибридная функция поиска, автоматическая декомпозиция вопроса и многомерный поиск:
        1. Декомпозиция вопроса на подвопросы с помощью LLM
        2. Семантический поиск по каждому подвопросу
        3. Извлечение связанных сущностей и получение их детальной информации
        4. Отслеживание цепочек связей
        5. Интеграция всех результатов, генерация глубокого анализа
        
        Args:
            graph_id: ID графа
            query: Вопрос пользователя
            simulation_requirement: Описание требований симуляции
            report_context: Контекст отчёта (необязательно, для более точной генерации подвопросов)
            max_sub_queries: Максимальное количество подвопросов
            
        Returns:
            InsightForgeResult: Результат глубокого анализа
        """
        logger.info(f"InsightForge глубокий анализ: {query[:50]}...")
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Генерация подвопросов с помощью LLM
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Сгенерировано {len(sub_queries)} подвопросов")
        
        # Step 2: Семантический поиск по каждому подвопросу
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # Поиск по исходному вопросу тоже
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Извлечение UUID связанных сущностей из рёбер, получение только их информации (без всех узлов)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Получение деталей всех связанных сущностей (без ограничения количества, полный вывод)
        entity_insights = []
        node_map = {}  # Для построения цепочек связей далее
        
        for uuid in list(entity_uuids):  # Обработка всех сущностей, без обрезки
            if not uuid:
                continue
            try:
                # Получение информации о каждом связанном узле
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Сущность")
                    
                    # Получение всех фактов, связанных с сущностью (без обрезки)
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # Полный вывод, без обрезки
                    })
            except Exception as e:
                logger.debug(f"Ошибка получения узла {uuid}: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Построение всех цепочек связей (без ограничения количества)
        relationship_chains = []
        for edge_data in all_edges:  # Обработка всех рёбер, без обрезки
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForge завершён: {result.total_facts} фактов, {result.total_entities} сущностей, {result.total_relationships} связей")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Генерация подвопросов с помощью LLM
        
        Декомпозиция сложного вопроса на несколько подвопросов для независимого поиска
        """
        system_prompt = """Ты профессиональный аналитик вопросов. Твоя задача — разложить сложный вопрос на несколько подвопросов, которые можно независимо наблюдать в мире симуляции.

Требования:
1. Каждый подвопрос должен быть достаточно конкретным, чтобы в мире симуляции можно было найти связанное поведение Agent или события
2. Подвопросы должны покрывать разные измерения исходного вопроса (кто, что, почему, как, когда, где)
3. Подвопросы должны быть связаны со сценарием симуляции
4. Формат ответа JSON: {"sub_queries": ["подвопрос1", "подвопрос2", ...]}"""

        user_prompt = f"""Контекст требований симуляции:
{simulation_requirement}

{f"Контекст отчёта: {report_context[:500]}" if report_context else ""}

Разложите следующий вопрос на {max_queries} подвопросов:
{query}

Верните список подвопросов в формате JSON."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # Убедиться что это список строк
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"Ошибка генерации подвопросов: {str(e)}, использование подвопросов по умолчанию")
            # Откат: варианты на основе исходного вопроса
            return [
                query,
                f"Основные участники {query}",
                f"Причины и последствия {query}",
                f"Процесс развития {query}"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch — широкий поиск】
        
        Получение полной картины, включая весь связанный контент и историческую/устаревшую информацию:
        1. Получение всех связанных узлов
        2. Получение всех рёбер (включая устаревшие/недействительные)
        3. Классификация текущей действующей и исторической информации
        
        Этот инструмент подходит для ситуаций, когда нужно понять полную картину события, отследить процесс эволюции.
        
        Args:
            graph_id: ID графа
            query: Поисковый запрос (для сортировки по релевантности)
            include_expired: Включать ли устаревший контент (по умолчанию True)
            limit: Лимит количества результатов
            
        Returns:
            PanoramaResult: Результат широкого поиска
        """
        logger.info(f"PanoramaSearch широкий поиск: {query[:50]}...")
        
        result = PanoramaResult(query=query)
        
        # Получение всех узлов
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Получение всех рёбер (с временной информацией)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # Классификация фактов
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # Добавление имён сущностей к фактам
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # Определение устаревания/недействительности
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # Исторические/устаревшие факты, добавление временной метки
                valid_at = edge.valid_at or "Неизвестно"
                invalid_at = edge.invalid_at or edge.expired_at or "Неизвестно"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Текущие действующие факты
                active_facts.append(edge.fact)
        
        # Сортировка по релевантности запроса
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Сортировка и ограничение количества
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"PanoramaSearch завершён: {result.active_count} действующих, {result.historical_count} исторических")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        【QuickSearch — простой поиск】
        
        Быстрый, лёгкий инструмент поиска:
        1. Прямой вызов семантического поиска Zep
        2. Возврат наиболее релевантных результатов
        3. Подходит для простых, прямых поисковых потребностей
        
        Args:
            graph_id: ID графа
            query: Поисковый запрос
            limit: Количество результатов
            
        Returns:
            SearchResult: Результат поиска
        """
        logger.info(f"QuickSearch простой поиск: {query[:50]}...")
        
        # Прямой вызов существующего метода search_graph
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearch завершён: {result.total_count} результатов")
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        【InterviewAgents — глубинное интервью】
        
        Вызов реального OASIS API для интервьюирования запущенных Agent симуляции:
        1. Автоматическое чтение файлов профилей для ознакомления со всеми Agent
        2. Анализ потребностей интервью с помощью LLM, интеллектуальный выбор наиболее релевантных Agent
        3. Генерация вопросов интервью с помощью LLM
        4. Вызов /api/simulation/interview/batch для реального интервью (одновременно на обеих платформах)
        5. Интеграция всех результатов, генерация отчёта интервью
        
        【Важно】Для этой функции требуется работающая среда симуляции (OASIS не закрыт)
        
        【Сценарии использования】
        - Нужно узнать мнение о событии с точки зрения различных ролей
        - Нужно собрать мнения и точки зрения разных сторон
        - Нужно получить реальные ответы Agent симуляции (не симуляция LLM)
        
        Args:
            simulation_id: ID симуляции (для нахождения файлов профилей и вызова API интервью)
            interview_requirement: Описание потребности интервью (неструктурированное, напр. "узнать мнение студентов о событии")
            simulation_requirement: Контекст требований симуляции (необязательно)
            max_agents: Максимальное количество Agent для интервью
            custom_questions: Пользовательские вопросы (необязательно, при отсутствии — автогенерация)
            
        Returns:
            InterviewResult: Результат интервью
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(f"InterviewAgents глубинное интервью (реальный API): {interview_requirement[:50]}...")
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Чтение файлов профилей
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(f"Файлы профилей для симуляции {simulation_id} не найдены")
            result.summary = "Файлы профилей Agent для интервью не найдены"
            return result
        
        result.total_agents = len(profiles)
        logger.info(f"Загружено {len(profiles)} профилей Agent")
        
        # Step 2: Выбор Agent для интервью с помощью LLM (возврат списка agent_id)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"Выбрано {len(selected_agents)} Agent для интервью: {selected_indices}")
        
        # Step 3: Генерация вопросов интервью (если не предоставлены)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"Сгенерировано {len(result.interview_questions)} вопросов интервью")
        
        # Объединение вопросов в один prompt для интервью
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # Добавление оптимизирующего префикса для ограничения формата ответа Agent
        INTERVIEW_PROMPT_PREFIX = (
            "Ты проходишь интервью. Основываясь на своём персонаже, всех прошлых воспоминаниях и действиях, "
            "ответь на следующие вопросы напрямую простым текстом.\n"
            "Требования к ответу:\n"
            "1. Отвечай на естественном языке, не вызывай никаких инструментов\n"
            "2. Не возвращай JSON или формат вызова инструментов\n"
            "3. Не используй заголовки Markdown (такие как #, ##, ###)\n"
            "4. Отвечай по номерам вопросов, каждый ответ начинай с «Вопрос X:» (X — номер вопроса)\n"
            "5. Разделяй ответы на вопросы пустой строкой\n"
            "6. Ответы должны быть содержательными, минимум 2-3 предложения на каждый вопрос\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: Вызов реального API интервью (без указания platform — по умолчанию обе платформы)
        try:
            # Построение списка пакетного интервью (без указания platform — обе платформы)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # Использование оптимизированного prompt
                    # Без указания platform — API интервьюирует на обеих платформах twitter и reddit
                })
            
            logger.info(f"Вызов пакетного API интервью (обе платформы): {len(interviews_request)} Agent")
            
            # Вызов метода пакетного интервью SimulationRunner (без platform — обе платформы)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # Без указания platform — обе платформы
                timeout=180.0   # Обе платформы требуют большего таймаута
            )
            
            logger.info(f"API интервью вернул: {api_result.get('interviews_count', 0)} результатов, success={api_result.get('success')}")
            
            # Проверка успешности вызова API
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Неизвестная ошибка")
                logger.warning(f"API интервью вернул ошибку: {error_msg}")
                result.summary = f"Ошибка вызова API интервью: {error_msg}. Проверьте состояние среды симуляции OASIS."
                return result
            
            # Step 5: Парсинг результатов API, построение объектов AgentInterview
            # Формат ответа двухплатформенного режима: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Неизвестно")
                agent_bio = agent.get("bio", "")
                
                # Получение результатов интервью Agent на обеих платформах
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Очистка возможной JSON-обёртки вызова инструментов
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Всегда выводить метки обеих платформ
                twitter_text = twitter_response if twitter_response else "(Ответ с этой платформы не получен)"
                reddit_text = reddit_response if reddit_response else "(Ответ с этой платформы не получен)"
                response_text = f"【Ответ на платформе Twitter】\n{twitter_text}\n\n【Ответ на платформе Reddit】\n{reddit_text}"

                # Извлечение ключевых цитат (из ответов обеих платформ)
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Очистка текста ответа: удаление меток, нумерации, Markdown и прочих помех
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                # Стратегия 1 (основная): извлечение полных содержательных предложений
                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # Стратегия 2 (дополнительная): длинный текст в правильно сопряжённых китайских кавычках「」
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # Расширенный лимит длины bio
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Среда симуляции не запущена
            logger.warning(f"Ошибка вызова API интервью (среда не запущена?): {e}")
            result.summary = f"Ошибка интервью: {str(e)}. Среда симуляции возможно закрыта, убедитесь что OASIS запущен."
            return result
        except Exception as e:
            logger.error(f"Исключение при вызове API интервью: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Ошибка в процессе интервью: {str(e)}"
            return result
        
        # Step 6: Генерация резюме интервью
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(f"InterviewAgents завершён: проинтервьюировано {result.interviewed_count} Agent (обе платформы)")
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Очистка JSON-обёртки вызова инструментов в ответе Agent, извлечение фактического содержимого"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Загрузка файлов профилей Agent симуляции"""
        import os
        import csv
        
        # Построение пути к файлам профилей
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # Приоритетная попытка чтения формата Reddit JSON
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"Загружено {len(profiles)} профилей из reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Ошибка чтения reddit_profiles.json: {e}")
        
        # Попытка чтения формата Twitter CSV
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Преобразование формата CSV в единый формат
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Неизвестно"
                        })
                logger.info(f"Загружено {len(profiles)} профилей из twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Ошибка чтения twitter_profiles.csv: {e}")
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        Выбор Agent для интервью с помощью LLM
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: Полная информация о выбранных Agent
                - selected_indices: Список индексов выбранных Agent (для вызова API)
                - reasoning: Обоснование выбора
        """
        
        # Построение списка резюме Agent
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Неизвестно"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """Ты профессиональный планировщик интервью. Твоя задача — по требованиям интервью выбрать из списка Agent симуляции наиболее подходящих для интервью.

Критерии выбора:
1. Идентичность/профессия Agent связана с темой интервью
2. Agent может иметь уникальную или ценную точку зрения
3. Выбирать разнообразные ракурсы (напр.: сторонники, противники, нейтральные, специалисты и т.д.)
4. Приоритет ролям, непосредственно связанным с событием

Формат ответа JSON:
{
    "selected_indices": [список индексов выбранных Agent],
    "reasoning": "пояснение выбора"
}"""

        user_prompt = f"""Требования интервью:
{interview_requirement}

Контекст симуляции:
{simulation_requirement if simulation_requirement else "Не предоставлен"}

Список доступных Agent (всего {len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Выберите максимум {max_agents} наиболее подходящих для интервью Agent и поясните выбор."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Автоматический выбор по релевантности")
            
            # Получение полной информации о выбранных Agent
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(f"Ошибка выбора Agent через LLM, использование выбора по умолчанию: {e}")
            # Откат: выбор первых N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Стратегия выбора по умолчанию"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Генерация вопросов интервью с помощью LLM"""
        
        agent_roles = [a.get("profession", "Неизвестно") for a in selected_agents]
        
        system_prompt = """Ты профессиональный журналист/интервьюер. Сгенерируй 3-5 глубинных вопросов для интервью на основе требований.

Требования к вопросам:
1. Открытые вопросы, поощряющие подробные ответы
2. Разные роли могут дать разные ответы
3. Покрытие нескольких измерений: факты, мнения, чувства и т.д.
4. Естественный язык, как в реальном интервью
5. Каждый вопрос — не более 50 слов, кратко и ясно
6. Прямой вопрос, без фоновых пояснений или префиксов

Формат ответа JSON: {"questions": ["вопрос1", "вопрос2", ...]}"""

        user_prompt = f"""Требования интервью: {interview_requirement}

Контекст симуляции: {simulation_requirement if simulation_requirement else "Не предоставлен"}

Роли интервьюируемых: {', '.join(agent_roles)}

Сгенерируйте 3-5 вопросов для интервью."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"Каково ваше мнение о {interview_requirement}?"])
            
        except Exception as e:
            logger.warning(f"Ошибка генерации вопросов интервью: {e}")
            return [
                f"Каково ваше мнение о {interview_requirement}?",
                "Как это влияет на вас или группу, которую вы представляете?",
                "Как, по вашему мнению, следует решить или улучшить эту проблему?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Генерация резюме интервью"""
        
        if not interviews:
            return "Ни одного интервью не проведено"
        
        # Сбор всего содержимого интервью
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}")
        
        system_prompt = """Ты профессиональный редактор новостей. Сгенерируй резюме интервью на основе ответов нескольких респондентов.

Требования к резюме:
1. Выделить основные точки зрения каждой стороны
2. Указать консенсус и расхождения во мнениях
3. Подчеркнуть ценные цитаты
4. Объективно и нейтрально, без предвзятости
5. Не более 1000 слов

Ограничения формата (обязательны):
- Использовать абзацы простого текста, разделять части пустыми строками
- Не использовать заголовки Markdown (такие как #, ##, ###)
- Не использовать разделительные линии (такие как ---, ***)
- При цитировании оригинальных слов респондентов использовать кавычки «»
- Можно использовать **жирный** для выделения ключевых слов, но не использовать другой синтаксис Markdown"""

        user_prompt = f"""Тема интервью: {interview_requirement}

Содержимое интервью:
{"".join(interview_texts)}

Сгенерируйте резюме интервью."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(f"Ошибка генерации резюме интервью: {e}")
            # Откат: простая конкатенация
            return f"Всего проинтервьюировано {len(interviews)} респондентов, включая: " + ", ".join([i.agent_name for i in interviews])
