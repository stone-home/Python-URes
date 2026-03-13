# Module Documentation (Deep) – Design

**Goal:** Produce module docs that go from basics through UML to full code understanding, for all logical modules in the project.

## Depth levels

1. **Basics** – Terminology, concepts, entry points. Enough for a new reader to know what the module is and where to start.
2. **Architecture & logic** – Core algorithms, workflows, dependencies (per existing rule).
3. **UML & structure** – Class diagrams, sequence diagrams where useful. Use Mermaid in-doc and reference existing PlantUML under `docs/PlantUML/`.
4. **Code-level understanding** – Main types, control flow, how components interact so a reader can follow the entire code path.
5. **Usage, API, maintenance** – Per existing module-documentation rule (integration, snippets, public API, troubleshooting).

## Document structure (extended template)

Every module doc under `docs/modules/` will follow:

- Frontmatter (title, last_updated, status, tags)
- 1. Overview
- 2. Basics (terminology, concepts, entry points)
- 3. Architecture & Logic
- 4. UML & Structure (class/sequence, Mermaid + PlantUML refs)
- 5. Code-Level Understanding (control flow, key types, integration points)
- 6. Usage & Examples
- 7. Public API / Interfaces
- 8. Maintenance & Troubleshooting
- 9. Execution Protocol (meta for writers)

## Modules in scope

- Memory (blocks, allocator, simulator)
- Data structures (tree, bi-directional links)
- Tools (enum, decorator)
- Core utilities (files, timedate, string, secrets, network)
- Literature search (search, adapters, paper, CLI)
- Literature citation (extractors, middlewares, rules, manager)
- Markdown (zettelkasten, manipulator)
- Docker (conf, image, container(s), runtime, cleanup)

## Deliverables

- One `module - <module-name>.md` per logical module in `docs/modules/`.
- Diagrams: Mermaid in Markdown for portability; cross-reference `docs/PlantUML/` where existing.
- No sensitive keys or hardcoded paths in docs.
