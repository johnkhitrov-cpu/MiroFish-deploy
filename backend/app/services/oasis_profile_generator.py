"""
Генератор OASIS Agent Profile
Преобразует сущности из графа Zep в формат Agent Profile, необходимый для платформы симуляции OASIS

Оптимизации:
1. Вызов функции поиска Zep для вторичного обогащения информации о узлах
2. Оптимизация промптов для генерации очень детальных персонажей
3. Разделение на персональные сущности и абстрактные групповые сущности
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """Структура данных OASIS Agent Profile"""
    # Общие поля
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str
    
    # Необязательные поля - стиль Reddit
    karma: int = 1000
    
    # Необязательные поля - стиль Twitter
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500
    
    # Дополнительная информация о персонаже
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)
    
    # Информация об исходной сущности
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def to_reddit_format(self) -> Dict[str, Any]:
        """Преобразование в формат платформы Reddit"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # Библиотека OASIS требует имя поля username (без подчёркивания)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }
        
        # Добавление дополнительной информации о персонаже (если есть)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """Преобразование в формат платформы Twitter"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # Библиотека OASIS требует имя поля username (без подчёркивания)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }
        
        # Добавление дополнительной информации о персонаже
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в полный формат словаря"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    Генератор OASIS Profile
    
    Преобразует сущности из графа Zep в Agent Profile, необходимые для симуляции OASIS
    
    Особенности оптимизации:
    1. Вызов функции поиска по графу Zep для получения более богатого контекста
    2. Генерация очень детальных персонажей (включая базовую информацию, карьеру, характер, поведение в соцсетях и т.д.)
    3. Разделение на персональные сущности и абстрактные групповые сущности
    """
    
    # Список типов MBTI
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    # Список распространённых стран
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France", 
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]
    
    # Типы персональных сущностей (требуется генерация конкретного персонажа)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure", 
        "expert", "faculty", "official", "journalist", "activist"
    ]
    
    # Типы групповых/организационных сущностей (требуется генерация представительного персонажа)
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo", 
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY не настроен")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Клиент Zep для получения расширенного контекста
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id
        
        if self.zep_api_key:
            try:
                self.zep_client = Zep(api_key=self.zep_api_key)
            except Exception as e:
                logger.warning(f"Не удалось инициализировать клиент Zep: {e}")
    
    def generate_profile_from_entity(
        self, 
        entity: EntityNode, 
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Генерация OASIS Agent Profile из сущности Zep
        
        Args:
            entity: Узел сущности Zep
            user_id: ID пользователя (для OASIS)
            use_llm: Использовать ли LLM для генерации детального персонажа
            
        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"
        
        # Базовая информация
        name = entity.name
        user_name = self._generate_username(name)
        
        # Построение контекстной информации
        context = self._build_entity_context(entity)
        
        if use_llm:
            # Использование LLM для генерации детального персонажа
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Использование правил для генерации базового персонажа
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Генерация имени пользователя"""
        # Удаление спецсимволов, приведение к нижнему регистру
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        
        # Добавление случайного суффикса для избежания дубликатов
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Использование функции гибридного поиска по графу Zep для получения расширенной информации о сущности
        
        Zep не имеет встроенного интерфейса гибридного поиска, поэтому нужно отдельно искать edges и nodes, затем объединять результаты.
        Используется параллельное выполнение запросов для повышения эффективности.
        
        Args:
            entity: Объект узла сущности
            
        Returns:
            Словарь с facts, node_summaries, context
        """
        import concurrent.futures
        
        if not self.zep_client:
            return {"facts": [], "node_summaries": [], "context": ""}
        
        entity_name = entity.name
        
        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }
        
        # Для поиска обязательно нужен graph_id
        if not self.graph_id:
            logger.debug(f"Пропуск поиска Zep: graph_id не задан")
            return results
        
        comprehensive_query = f"Вся информация, активность, события, связи и фон о {entity_name}"
        
        def search_edges():
            """Поиск рёбер (фактов/связей) — с механизмом повторных попыток"""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Поиск рёбер Zep, попытка {attempt + 1} не удалась: {str(e)[:80]}, повтор...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Поиск рёбер Zep не удался после {max_retries} попыток: {e}")
            return None
        
        def search_nodes():
            """Поиск узлов (описаний сущностей) — с механизмом повторных попыток"""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Поиск узлов Zep, попытка {attempt + 1} не удалась: {str(e)[:80]}, повтор...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Поиск узлов Zep не удался после {max_retries} попыток: {e}")
            return None
        
        try:
            # Параллельное выполнение поиска по edges и nodes
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)
                
                # Получение результатов
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)
            
            # Обработка результатов поиска рёбер
            all_facts = set()
            if edge_result and hasattr(edge_result, 'edges') and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)
            
            # Обработка результатов поиска узлов
            all_summaries = set()
            if node_result and hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, 'name') and node.name and node.name != entity_name:
                        all_summaries.add(f"Связанная сущность: {node.name}")
            results["node_summaries"] = list(all_summaries)
            
            # Построение комплексного контекста
            context_parts = []
            if results["facts"]:
                context_parts.append("Фактическая информация:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Связанные сущности:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)
            
            logger.info(f"Гибридный поиск Zep завершён: {entity_name}, получено {len(results['facts'])} фактов, {len(results['node_summaries'])} связанных узлов")
            
        except concurrent.futures.TimeoutError:
            logger.warning(f"Таймаут поиска Zep ({entity_name})")
        except Exception as e:
            logger.warning(f"Ошибка поиска Zep ({entity_name}): {e}")
        
        return results
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Построение полной контекстной информации о сущности
        
        Включает:
        1. Информацию о рёбрах сущности (факты)
        2. Детальную информацию о связанных узлах
        3. Расширенную информацию из гибридного поиска Zep
        """
        context_parts = []
        
        # 1. Добавление информации об атрибутах сущности
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Атрибуты сущности\n" + "\n".join(attrs))
        
        # 2. Добавление информации о связанных рёбрах (факты/связи)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # Без ограничения количества
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")
                
                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (связанная сущность)")
                    else:
                        relationships.append(f"- (связанная сущность) --[{edge_name}]--> {entity.name}")
            
            if relationships:
                context_parts.append("### Связанные факты и отношения\n" + "\n".join(relationships))
        
        # 3. Добавление детальной информации о связанных узлах
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # Без ограничения количества
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")
                
                # Фильтрация меток по умолчанию
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""
                
                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")
            
            if related_info:
                context_parts.append("### Информация о связанных сущностях\n" + "\n".join(related_info))
        
        # 4. Использование гибридного поиска Zep для получения расширенной информации
        zep_results = self._search_zep_for_entity(entity)
        
        if zep_results.get("facts"):
            # Дедупликация: исключение уже существующих фактов
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Фактическая информация из поиска Zep\n" + "\n".join(f"- {f}" for f in new_facts[:15]))
        
        if zep_results.get("node_summaries"):
            context_parts.append("### Связанные узлы из поиска Zep\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))
        
        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """Определение, является ли сущность персональной"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES
    
    def _is_group_entity(self, entity_type: str) -> bool:
        """Определение, является ли сущность групповой/организационной"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Генерация очень детального персонажа с помощью LLM
        
        В зависимости от типа сущности:
        - Персональная сущность: генерация конкретного персонажа
        - Групповая/организационная сущность: генерация представительного аккаунта
        """
        
        is_individual = self._is_individual_entity(entity_type)
        
        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Попытка генерации с повторами до достижения максимального количества попыток
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Снижение температуры при каждой повторной попытке
                    # Не устанавливаем max_tokens, даём LLM свободу
                )
                
                content = response.choices[0].message.content
                
                # Проверка, не был ли вывод обрезан (finish_reason не 'stop')
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"Вывод LLM обрезан (попытка {attempt+1}), попытка исправления...")
                    content = self._fix_truncated_json(content)
                
                # Попытка парсинга JSON
                try:
                    result = json.loads(content)
                    
                    # Валидация обязательных полей
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} — это {entity_type}."
                    
                    return result
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"Ошибка парсинга JSON (попытка {attempt+1}): {str(je)[:80]}")
                    
                    # Попытка исправления JSON
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result
                    
                    last_error = je
                    
            except Exception as e:
                logger.warning(f"Ошибка вызова LLM (попытка {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # Экспоненциальная задержка
        
        logger.warning(f"Генерация персонажа через LLM не удалась ({max_attempts} попыток): {last_error}, используется генерация по правилам")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Исправление обрезанного JSON (вывод обрезан ограничением max_tokens)"""
        import re
        
        # Если JSON обрезан, попытка его закрыть
        content = content.strip()
        
        # Подсчёт незакрытых скобок
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Проверка, есть ли незакрытая строка
        # Простая проверка: если после последней кавычки нет запятой или закрывающей скобки, возможно строка обрезана
        if content and content[-1] not in '",}]':
            # Попытка закрыть строку
            content += '"'
        
        # Закрытие скобок
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Попытка исправления повреждённого JSON"""
        import re
        
        # 1. Сначала попытка исправить обрезанный вывод
        content = self._fix_truncated_json(content)
        
        # 2. Попытка извлечения JSON-части
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 3. Обработка переносов строк внутри строковых значений
            # Поиск всех строковых значений и замена переносов в них
            def fix_string_newlines(match):
                s = match.group(0)
                # Замена реальных переносов строк на пробелы
                s = s.replace('\n', ' ').replace('\r', ' ')
                # Замена лишних пробелов
                s = re.sub(r'\s+', ' ', s)
                return s
            
            # Поиск строковых значений JSON
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)
            
            # 4. Попытка парсинга
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Если всё ещё не удалось, более агрессивное исправление
                try:
                    # Удаление всех управляющих символов
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # Замена всех последовательных пробелов
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass
        
        # 6. Попытка извлечения частичной информации из содержимого
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # Может быть обрезано
        
        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} — это {entity_type}.")
        
        # Если удалось извлечь значимую информацию, помечаем как исправленное
        if bio_match or persona_match:
            logger.info(f"Извлечена частичная информация из повреждённого JSON")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }
        
        # 7. Полная неудача, возврат базовой структуры
        logger.warning(f"Исправление JSON не удалось, возврат базовой структуры")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} — это {entity_type}."
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """Получение системного промпта"""
        base_prompt = "Вы — эксперт по созданию профилей пользователей социальных сетей. Генерируйте детальные, реалистичные персонажи для симуляции общественного мнения, максимально воспроизводя реальную ситуацию. Необходимо вернуть валидный JSON-формат, все строковые значения не должны содержать неэкранированных переносов строк. Используйте русский язык."
        return base_prompt
    
    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Построение промпта для детального персонажа индивидуальной сущности"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Нет"
        context_str = context[:3000] if context else "Нет дополнительного контекста"
        
        return f"""Сгенерируйте детальный персонаж пользователя социальных сетей для сущности, максимально воспроизводя реальную ситуацию.

Название сущности: {entity_name}
Тип сущности: {entity_type}
Описание сущности: {entity_summary}
Атрибуты сущности: {attrs_str}

Контекстная информация:
{context_str}

Сгенерируйте JSON, содержащий следующие поля:

1. bio: Описание профиля в соцсети, 200 символов
2. persona: Детальное описание персонажа (2000 символов чистого текста), должно включать:
   - Базовая информация (возраст, профессия, образование, местоположение)
   - Фон персонажа (важные события, связь с событиями, социальные связи)
   - Черты характера (тип MBTI, ключевые черты, способ выражения эмоций)
   - Поведение в соцсетях (частота публикаций, предпочтения по контенту, стиль общения, языковые особенности)
   - Позиция и взгляды (отношение к теме, что может вызвать раздражение/сочувствие)
   - Уникальные черты (крылатые фразы, особый опыт, личные хобби)
   - Личная память (важная часть персонажа, описание связи данной персоны с событием, а также её действия и реакции в контексте события)
3. age: Возраст числом (обязательно целое число)
4. gender: Пол, обязательно на английском: "male" или "female"
5. mbti: Тип MBTI (например INTJ, ENFP и т.д.)
6. country: Страна (на русском, например "Россия")
7. profession: Профессия
8. interested_topics: Массив интересующих тем

Важно:
- Все значения полей должны быть строками или числами, не используйте переносы строк
- persona должно быть связным текстовым описанием
- Используйте русский язык (кроме поля gender, которое должно быть на английском male/female)
- Содержание должно быть согласовано с информацией о сущности
- age должен быть валидным целым числом, gender должен быть "male" или "female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Построение промпта для детального персонажа групповой/организационной сущности"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Нет"
        context_str = context[:3000] if context else "Нет дополнительного контекста"
        
        return f"""Сгенерируйте детальное описание аккаунта в социальных сетях для организации/группы, максимально воспроизводя реальную ситуацию.

Название сущности: {entity_name}
Тип сущности: {entity_type}
Описание сущности: {entity_summary}
Атрибуты сущности: {attrs_str}

Контекстная информация:
{context_str}

Сгенерируйте JSON, содержащий следующие поля:

1. bio: Описание официального аккаунта, 200 символов, профессионально и уместно
2. persona: Детальное описание аккаунта (2000 символов чистого текста), должно включать:
   - Базовая информация об организации (официальное название, тип организации, история создания, основные функции)
   - Позиционирование аккаунта (тип аккаунта, целевая аудитория, основные функции)
   - Стиль высказываний (языковые особенности, типичные выражения, запретные темы)
   - Особенности публикуемого контента (типы контента, частота публикаций, активные часы)
   - Позиция и отношение (официальная позиция по ключевым темам, способ реагирования на споры)
   - Специальные примечания (портрет представляемой группы, привычки ведения аккаунта)
   - Память организации (важная часть персонажа организации, описание связи организации с событием, а также её действия и реакции в контексте события)
3. age: Фиксированное значение 30 (виртуальный возраст аккаунта организации)
4. gender: Фиксированное значение "other" (для аккаунтов организаций используется other, т.к. это не личный аккаунт)
5. mbti: Тип MBTI для описания стиля аккаунта, например ISTJ для строгого и консервативного
6. country: Страна (на русском, например "Россия")
7. profession: Описание функций организации
8. interested_topics: Массив областей интересов

Важно:
- Все значения полей должны быть строками или числами, null-значения не допускаются
- persona должно быть связным текстовым описанием, без переносов строк
- Используйте русский язык (кроме поля gender, которое должно быть на английском "other")
- age должен быть целым числом 30, gender должен быть строкой "other"
- Высказывания аккаунта организации должны соответствовать её позиционированию"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Генерация базового персонажа на основе правил"""
        
        # Генерация разного персонажа в зависимости от типа сущности
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # Виртуальный возраст организации
                "gender": "other",  # Для организаций используется other
                "mbti": "ISTJ",  # Стиль организации: строгий и консервативный
                "country": "Китай",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }
        
        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # Виртуальный возраст организации
                "gender": "other",  # Для организаций используется other
                "mbti": "ISTJ",  # Стиль организации: строгий и консервативный
                "country": "Китай",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }
        
        else:
            # Персонаж по умолчанию
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Установка ID графа для поиска по Zep"""
        self.graph_id = graph_id
    
    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """
        Пакетная генерация Agent Profile из сущностей (с поддержкой параллельной генерации)
        
        Args:
            entities: Список сущностей
            use_llm: Использовать ли LLM для генерации детальных персонажей
            progress_callback: Функция обратного вызова для прогресса (current, total, message)
            graph_id: ID графа для поиска Zep для получения расширенного контекста
            parallel_count: Количество параллельных генераций, по умолчанию 5
            realtime_output_path: Путь файла для записи в реальном времени (если указан, каждый сгенерированный профиль записывается сразу)
            output_platform: Формат выходной платформы ("reddit" или "twitter")
            
        Returns:
            Список Agent Profile
        """
        import concurrent.futures
        from threading import Lock
        
        # Установка graph_id для поиска Zep
        if graph_id:
            self.graph_id = graph_id
        
        total = len(entities)
        profiles = [None] * total  # Предварительное выделение списка для сохранения порядка
        completed_count = [0]  # Использование списка для возможности изменения в замыкании
        lock = Lock()
        
        # Вспомогательная функция для записи в файл в реальном времени
        def save_profiles_realtime():
            """Сохранение сгенерированных профилей в файл в реальном времени"""
            if not realtime_output_path:
                return
            
            with lock:
                # Фильтрация уже сгенерированных профилей
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return
                
                try:
                    if output_platform == "reddit":
                        # Формат Reddit JSON
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Формат Twitter CSV
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Не удалось сохранить профили в реальном времени: {e}")
        
        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Рабочая функция генерации одного профиля"""
            entity_type = entity.get_entity_type() or "Entity"
            
            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )
                
                # Вывод сгенерированного персонажа в консоль и лог в реальном времени
                self._print_generated_profile(entity.name, entity_type, profile)
                
                return idx, profile, None
                
            except Exception as e:
                logger.error(f"Не удалось сгенерировать персонаж для сущности {entity.name}: {str(e)}")
                # Создание базового профиля
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)
        
        logger.info(f"Начало параллельной генерации {total} персонажей Agent (параллельность: {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Начало генерации персонажей Agent — всего {total} сущностей, параллельность: {parallel_count}")
        print(f"{'='*60}\n")
        
        # Параллельное выполнение с помощью пула потоков
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Отправка всех задач
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }
            
            # Сбор результатов
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"
                
                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile
                    
                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]
                    
                    # Запись в файл в реальном времени
                    save_profiles_realtime()
                    
                    if progress_callback:
                        progress_callback(
                            current, 
                            total, 
                            f"Завершено {current}/{total}: {entity.name} ({entity_type})"
                        )
                    
                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} использует резервный персонаж: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Персонаж успешно сгенерирован: {entity.name} ({entity_type})")
                        
                except Exception as e:
                    logger.error(f"Исключение при обработке сущности {entity.name}: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Запись в файл в реальном времени (даже для резервных персонажей)
                    save_profiles_realtime()
        
        print(f"\n{'='*60}")
        print(f"Генерация персонажей завершена! Сгенерировано {len([p for p in profiles if p])} Agent")
        print(f"{'='*60}\n")
        
        return profiles
    
    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Вывод сгенерированного персонажа в консоль в реальном времени (полное содержимое, без обрезки)"""
        separator = "-" * 70
        
        # Построение полного вывода (без обрезки)
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'Нет'
        
        output_lines = [
            f"\n{separator}",
            f"[Сгенерировано] {entity_name} ({entity_type})",
            f"{separator}",
            f"Имя пользователя: {profile.user_name}",
            f"",
            f"【Описание】",
            f"{profile.bio}",
            f"",
            f"【Детальный персонаж】",
            f"{profile.persona}",
            f"",
            f"【Базовые атрибуты】",
            f"Возраст: {profile.age} | Пол: {profile.gender} | MBTI: {profile.mbti}",
            f"Профессия: {profile.profession} | Страна: {profile.country}",
            f"Интересующие темы: {topics_str}",
            separator
        ]
        
        output = "\n".join(output_lines)
        
        # Вывод только в консоль (чтобы избежать дублирования, logger не выводит полное содержимое)
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        Сохранение Profile в файл (с выбором правильного формата в зависимости от платформы)
        
        Требования к формату платформ OASIS:
        - Twitter: формат CSV
        - Reddit: формат JSON
        
        Args:
            profiles: Список Profile
            file_path: Путь к файлу
            platform: Тип платформы ("reddit" или "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Сохранение Twitter Profile в формате CSV (соответствует официальным требованиям OASIS)
        
        Поля CSV, требуемые OASIS Twitter:
        - user_id: ID пользователя (начинается с 0 по порядку в CSV)
        - name: Настоящее имя пользователя
        - username: Имя пользователя в системе
        - user_char: Детальное описание персонажа (внедряется в системный промпт LLM, управляет поведением Agent)
        - description: Краткое публичное описание (отображается на странице профиля)
        
        Разница между user_char и description:
        - user_char: Для внутреннего использования, системный промпт LLM, определяет как Agent думает и действует
        - description: Для внешнего отображения, видно другим пользователям
        """
        import csv
        
        # Проверка расширения файла (.csv)
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Запись заголовков, требуемых OASIS
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)
            
            # Запись строк данных
            for idx, profile in enumerate(profiles):
                # user_char: Полный персонаж (bio + persona), для системного промпта LLM
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Обработка переносов строк (в CSV заменяются пробелами)
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')
                
                # description: Краткое описание для внешнего отображения
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    idx,                    # user_id: последовательный ID начиная с 0
                    profile.name,           # name: настоящее имя
                    profile.user_name,      # username: имя пользователя
                    user_char,              # user_char: полный персонаж (для внутреннего LLM)
                    description             # description: краткое описание (для внешнего отображения)
                ]
                writer.writerow(row)
        
        logger.info(f"Сохранено {len(profiles)} Twitter Profile в {file_path} (формат OASIS CSV)")
    
    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Нормализация поля gender к формату, требуемому OASIS (английский)
        
        OASIS требует: male, female, other
        """
        if not gender:
            return "other"
        
        gender_lower = gender.lower().strip()
        
        # Маппинг
        gender_map = {
            "男": "male",
            "女": "female",
            "机构": "other",
            "其他": "other",
            "муж": "male",
            "жен": "female",
            "мужской": "male",
            "женский": "female",
            "организация": "other",
            "другое": "other",
            # Английские значения
            "male": "male",
            "female": "female",
            "other": "other",
        }
        
        return gender_map.get(gender_lower, "other")
    
    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Сохранение Reddit Profile в формате JSON
        
        Используется формат, совместимый с to_reddit_format(), для корректного чтения OASIS.
        Обязательно должно содержать поле user_id — это ключ для сопоставления в OASIS agent_graph.get_agent()!
        
        Обязательные поля:
        - user_id: ID пользователя (целое число, для сопоставления с poster_agent_id в initial_posts)
        - username: Имя пользователя
        - name: Отображаемое имя
        - bio: Описание
        - persona: Детальный персонаж
        - age: Возраст (целое число)
        - gender: "male", "female", или "other"
        - mbti: Тип MBTI
        - country: Страна
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Использование формата, совместимого с to_reddit_format()
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # Критично: обязательно содержать user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Обязательные поля OASIS — обеспечение значений по умолчанию
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "Китай",
            }
            
            # Необязательные поля
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics
            
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Сохранено {len(profiles)} Reddit Profile в {file_path} (формат JSON, содержит поле user_id)")
    
    # Сохранение старого имени метода как псевдонима для обратной совместимости
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Устарело] Используйте метод save_profiles()"""
        logger.warning("save_profiles_to_json устарел, используйте метод save_profiles")
        self.save_profiles(profiles, file_path, platform)
