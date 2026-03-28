"""
Сервис генерации онтологий
Интерфейс 1: Анализ текстового содержимого, генерация определений типов сущностей и связей для социальной симуляции
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# Системный промпт для генерации онтологий
ONTOLOGY_SYSTEM_PROMPT = """Вы — профессиональный эксперт по проектированию онтологий графов знаний. Ваша задача — проанализировать заданный текст и требования к симуляции, спроектировать типы сущностей и типы связей, подходящие для **симуляции общественного мнения в социальных сетях**.

**Важно: вы должны вывести данные в валидном JSON-формате, не выводите ничего другого.**

## Контекст основной задачи

Мы создаём **систему симуляции общественного мнения в социальных сетях**. В этой системе:
- Каждая сущность — это «аккаунт» или «субъект», который может высказываться, взаимодействовать, распространять информацию в соцсетях
- Сущности взаимно влияют друг на друга, делают репосты, комментируют, отвечают
- Нам нужно симулировать реакции сторон и пути распространения информации в медийных событиях

Поэтому **сущности должны быть реально существующими субъектами, способными высказываться и взаимодействовать в соцсетях**:

**Допустимы**:
- Конкретные люди (публичные личности, участники событий, лидеры мнений, эксперты и учёные, обычные люди)
- Компании, предприятия (включая их официальные аккаунты)
- Организации (университеты, ассоциации, НКО, профсоюзы и т.д.)
- Государственные органы, регуляторы
- Медиа-организации (газеты, телеканалы, блогеры, веб-сайты)
- Сами платформы социальных сетей
- Представители определённых групп (например, ассоциации выпускников, фан-клубы, правозащитные группы и т.д.)

**Недопустимы**:
- Абстрактные понятия (например, «общественное мнение», «эмоции», «тенденции»)
- Темы/топики (например, «академическая честность», «реформа образования»)
- Точки зрения/позиции (например, «сторонники», «оппоненты»)

## Формат вывода

Выведите JSON со следующей структурой:

```json
{
    "entity_types": [
        {
            "name": "Название типа сущности (англ., PascalCase)",
            "description": "Краткое описание (англ., не более 100 символов)",
            "attributes": [
                {
                    "name": "Имя атрибута (англ., snake_case)",
                    "type": "text",
                    "description": "Описание атрибута"
                }
            ],
            "examples": ["Пример сущности 1", "Пример сущности 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Название типа связи (англ., UPPER_SNAKE_CASE)",
            "description": "Краткое описание (англ., не более 100 символов)",
            "source_targets": [
                {"source": "Тип сущности-источника", "target": "Тип целевой сущности"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Краткий анализ текстового содержимого (на русском)"
}
```

## Руководство по проектированию (крайне важно!)

### 1. Проектирование типов сущностей — строго обязательно

**Требование к количеству: ровно 10 типов сущностей**

**Требования к иерархической структуре (должны присутствовать как конкретные, так и резервные типы)**:

Ваши 10 типов сущностей должны включать следующие уровни:

A. **Резервные типы (обязательны, размещаются последними 2 в списке)**:
   - `Person`: Резервный тип для любого физического лица. Когда человек не относится к другим, более конкретным типам людей, он попадает в эту категорию.
   - `Organization`: Резервный тип для любой организации. Когда организация не относится к другим, более конкретным типам организаций, она попадает в эту категорию.

B. **Конкретные типы (8, проектируются на основе содержания текста)**:
   - Проектируйте более конкретные типы для основных ролей, встречающихся в тексте
   - Например: если текст связан с академическими событиями, можно использовать `Student`, `Professor`, `University`
   - Например: если текст связан с бизнес-событиями, можно использовать `Company`, `CEO`, `Employee`

**Зачем нужны резервные типы**:
- В тексте встречаются различные персонажи, такие как «школьные учителя», «случайные прохожие», «некий пользователь сети»
- Если нет специального подходящего типа, они должны попадать в `Person`
- Аналогично, малые организации, временные группы и т.д. должны попадать в `Organization`

**Принципы проектирования конкретных типов**:
- Выявляйте часто встречающиеся или ключевые роли из текста
- Каждый конкретный тип должен иметь чёткие границы, избегая пересечений
- description должен чётко пояснять отличие этого типа от резервного

### 2. Проектирование типов связей

- Количество: 6-10
- Связи должны отражать реальные взаимосвязи в социальных сетях
- Убедитесь, что source_targets связей охватывают определённые вами типы сущностей

### 3. Проектирование атрибутов

- 1-3 ключевых атрибута для каждого типа сущности
- **Внимание**: в качестве имён атрибутов нельзя использовать `name`, `uuid`, `group_id`, `created_at`, `summary` (это системные зарезервированные слова)
- Рекомендуется использовать: `full_name`, `title`, `role`, `position`, `location`, `description` и т.д.

## Справочник типов сущностей

**Персональные (конкретные)**:
- Student: Студент
- Professor: Профессор/Учёный
- Journalist: Журналист
- Celebrity: Знаменитость/Блогер
- Executive: Топ-менеджер
- Official: Государственный чиновник
- Lawyer: Юрист
- Doctor: Врач

**Персональные (резервные)**:
- Person: Любое физическое лицо (используется, когда не подходит ни один из конкретных типов выше)

**Организационные (конкретные)**:
- University: Университет
- Company: Компания/Предприятие
- GovernmentAgency: Государственный орган
- MediaOutlet: Медиа-организация
- Hospital: Больница
- School: Школа
- NGO: Некоммерческая организация

**Организационные (резервные)**:
- Organization: Любая организация (используется, когда не подходит ни один из конкретных типов выше)

## Справочник типов связей

- WORKS_FOR: Работает в
- STUDIES_AT: Учится в
- AFFILIATED_WITH: Аффилирован с
- REPRESENTS: Представляет
- REGULATES: Регулирует
- REPORTS_ON: Освещает (в СМИ)
- COMMENTS_ON: Комментирует
- RESPONDS_TO: Отвечает на
- SUPPORTS: Поддерживает
- OPPOSES: Выступает против
- COLLABORATES_WITH: Сотрудничает с
- COMPETES_WITH: Конкурирует с
"""


class OntologyGenerator:
    """
    Генератор онтологий
    Анализирует текстовое содержимое, генерирует определения типов сущностей и связей
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Генерация определения онтологии

        Args:
            document_texts: Список текстов документов
            simulation_requirement: Описание требований к симуляции
            additional_context: Дополнительный контекст

        Returns:
            Определение онтологии (entity_types, edge_types и т.д.)
        """
        # Построение пользовательского сообщения
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Вызов LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Валидация и постобработка
        result = self._validate_and_process(result)

        return result

    # Максимальная длина текста для передачи LLM (50 тыс. символов)
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Построение пользовательского сообщения"""

        # Объединение текстов
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # Если текст превышает 50 тыс. символов, обрезаем (влияет только на передачу в LLM, не на построение графа)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(оригинал содержит {original_length} символов, взяты первые {self.MAX_TEXT_LENGTH_FOR_LLM} для анализа онтологии)..."

        message = f"""## Требования к симуляции

{simulation_requirement}

## Содержание документа

{combined_text}
"""

        if additional_context:
            message += f"""
## Дополнительные пояснения

{additional_context}
"""

        message += """
На основе вышеприведённого содержания спроектируйте типы сущностей и типы связей, подходящие для симуляции общественного мнения.

**Обязательные правила**:
1. Необходимо вывести ровно 10 типов сущностей
2. Последние 2 должны быть резервными типами: Person (резервный для людей) и Organization (резервный для организаций)
3. Первые 8 — конкретные типы, спроектированные на основе содержания текста
4. Все типы сущностей должны быть реальными субъектами, способными высказываться, а не абстрактными понятиями
5. В именах атрибутов нельзя использовать зарезервированные слова name, uuid, group_id и т.д., используйте full_name, org_name и т.п.
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация и постобработка результата"""

        # Проверка наличия обязательных полей
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Валидация типов сущностей
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Проверка, что description не превышает 100 символов
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Валидация типов связей
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Ограничение Zep API: максимум 10 пользовательских типов сущностей, максимум 10 пользовательских типов связей
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Определения резервных типов
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }

        # Проверка наличия резервных типов
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Резервные типы, которые нужно добавить
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # Если после добавления превысим 10, нужно удалить некоторые существующие типы
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Вычисляем сколько нужно удалить
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Удаляем с конца (сохраняем более важные конкретные типы в начале)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Добавляем резервные типы
            result["entity_types"].extend(fallbacks_to_add)

        # Финальная проверка на превышение лимита (защитное программирование)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Преобразование определения онтологии в Python-код (аналогично ontology.py)

        Args:
            ontology: Определение онтологии

        Returns:
            Строка с Python-кодом
        """
        code_lines = [
            '"""',
            'Определения пользовательских типов сущностей',
            'Автоматически сгенерировано MiroFish для симуляции общественного мнения',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Определения типов сущностей ==============',
            '',
        ]

        # Генерация типов сущностей
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== Определения типов связей ==============')
        code_lines.append('')

        # Генерация типов связей
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Преобразование в PascalCase имя класса
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        # Генерация словарей типов
        code_lines.append('# ============== Конфигурация типов ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        # Генерация маппинга source_targets для связей
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)
