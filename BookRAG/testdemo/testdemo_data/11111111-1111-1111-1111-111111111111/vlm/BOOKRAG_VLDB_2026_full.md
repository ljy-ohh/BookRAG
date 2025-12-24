# BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents

Shu Wang

The Chinese University of Hong

Kong, Shenzhen

shuwang3@link.cuhk.edu.cn

Yingli Zhou

The Chinese University of Hong

Kong, Shenzhen

yinglizhou@link.cuhk.edu.cn

Yixiang Fang

The Chinese University of Hong

Kong, Shenzhen

fangyixiang@cuhk.edu.cn

# ABSTRACT

As an effective method to boost the performance of Large Language Models (LLMs) on the question answering (QA) task, Retrieval-Augmented Generation (RAG), which queries highly relevant information from external complex documents, has attracted tremendous attention from both industry and academia. Existing RAG approaches often focus on general documents, and they overlook the fact that many real-world documents (such as books, booklets, handbooks, etc.) have a hierarchical structure, which organizes their content from different granularity levels, leading to poor performance for the QA task. To address these limitations, we introduce BookRAG, a novel RAG approach targeted for documents with a hierarchical structure, which exploits logical hierarchies and traces entity relations to query the highly relevant information. Specifically, we build a novel index structure, called BookIndex, by extracting a hierarchical tree from the document, which serves as the role of its table of contents, using a graph to capture the intricate relationships between entities, and mapping entities to tree nodes. Leveraging the BookIndex, we then propose an agent-based query method inspired by the Information Foraging Theory, which dynamically classifies queries and employs a tailored retrieval workflow. Extensive experiments on three widely adopted benchmarks demonstrate that BookRAG achieves state-of-the-art performance, significantly outperforming baselines in both retrieval recall and QA accuracy while maintaining competitive efficiency.

# PVLDB Reference Format:

Shu Wang, Yingli Zhou, and Yixiang Fang. BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents. PVLDB, 19(1): XXX-XXX, 2025. doi:XX.XX/XXX.XX

# PVLDB Artifact Availability:

The source code, data, and/or other artifacts have been made available at https://github.com/sam234990/BookRAG.

# 1 INTRODUCTION

Large Language Models (LLMs) such as Qwen 3 [60] and Gemini 2.5 [13] have revolutionized the Question Answering (QA) system [15, 61, 65]. The industry has increasingly adopted LLMs to build QA systems that assist users and reduce manual effort in

This work is licensed under the Creative Commons BY-NC-ND 4.0 International License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of this license. For any use beyond those covered by this license, obtain permission by emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights licensed to the VLDB Endowment.  
Proceedings of the VLDB Endowment, Vol. 19, No. 1 ISSN 2150-8097.  
doi:XX.XX/XXX.XX

![](images/f40a8e134ff67e3c0c657345ee1afc5bfa6d85014b2d7dfad9cfaa160f94c60c.jpg)  
Figure 1: Comparison of existing methods and BookRAG for complex document QA.

many applications [65, 67], such as financial auditing [29, 37], legal compliance [8], and scientific discovery [56]. However, directly relying on LLMs may lead to missing domain knowledge and generating outdated or unsupported information. To address these issues, Retrieval-Augmented Generation (RAG) has been widely adopted [17, 22] by retrieving relevant domain knowledge from external sources and using it to guide the LLM during response generation. On the other hand, in real-world enterprise scenarios, domain knowledge is often stored in long-form documents, such as technical handbooks, API reference manuals, and operational guidebooks [49]. A notable feature of such documents is that they follow the structure of books, characterized by intricate layouts and rigorous logical hierarchies (e.g., explicit tables of contents, nested chapters, and multi-level sections). In this paper, we aim to design an effective RAG system for QA over long and highly structured documents.

- Prior works. The existing RAG approaches for document-level QA generally fall into two paradigms, as illustrated in Figure 1. The first paradigm relies on OCR (Optical Character Recognition) to convert the document into plain text, after which any text-based RAG method can be directly applied. Among text-based RAG methods, state-of-the-art approaches increasingly adopt graph-based RAG [6, 62, 66], where graph data serves as an external knowledge source because it captures rich semantic information and the

Table 1: Comparison of representative methods and our BookRAG.  

<table><tr><td>Type</td><td>Representative Method</td><td>Core Feature</td><td>Multi-hop Reasoning</td><td>Document Parsing</td><td>Query Workflow</td></tr><tr><td rowspan="2">Graph-based</td><td>RAPTOR [45]</td><td>Recursive summarization</td><td>✓</td><td>✗</td><td>Static</td></tr><tr><td>GraphRAG [16]</td><td>Global community detection</td><td>✓</td><td>✗</td><td>Static</td></tr><tr><td rowspan="2">Layout segmented</td><td>MM-Vanilla</td><td>Multi-modal retrieval</td><td>✗</td><td>✓</td><td>Static</td></tr><tr><td>DocETL [47]</td><td>LLM-based document processing pipeline</td><td>✗</td><td>✓</td><td>Manual</td></tr><tr><td>Doc-Native</td><td>BookRAG (Ours)</td><td>Structure-award Index &amp; Agent-based retrieval</td><td>✓</td><td>✓</td><td>Dynamic</td></tr></table>

relational structure between entities. As shown in Table 1, two representative methods are GraphRAG [16] and RAPTOR [45]. Specifically, GraphRAG first constructs a knowledge graph (KG) from the textual corpus, and then applies the Leiden community detection algorithm [51] to obtain hierarchical clusters. Summaries are generated for each community, providing a comprehensive, global overview of the entire corpus. RAPTOR builds a recursive tree structure by iteratively clustering document chunks and summarizing them at each level, enabling the model to capture both fine-grained and high-level semantic information across the corpus.

In contrast, the second paradigm, layout-aware segmentation [5, 52], first parses the document into structured blocks that preserve the original layout and information of the document, such as paragraphs, tables, figures, or equations. By doing so, it not only avoids the fixed chunk size used in the first paradigm, which often leads to fragmented information, but also retains document-native structural information. These blocks often exhibit multimodal characteristics, and a typical approach is to apply multimodal retrieval to obtain relevant content for answering queries. Recently, a state-of-the-art method in this category, DocETL [47], provides a declarative interface that allows users to manually define LLM-based processing pipelines to analyze the retrieved blocks. These pipelines consist of LLM-powered operations combined with task-specific optimizations.

- Limitations of existing works. However, these methods suffer from two fundamental limitations (L for short): L1: Failure to capture the deep connection of document structure and semantics. Text-based approaches cannot capture the structural layout of the document, resulting in the loss of important relationships stored in the hierarchical blocks, such as tables nested within a specific section. While layout-segmented methods preserve document structure, they cannot capture the relationships between different blocks in the document, which limits their capability for multi-hop reasoning across these blocks and ultimately affects their overall performance. L2: Static of query workflows. In real-world QA scenarios, user queries are highly heterogeneous, ranging from simple keyword lookups to complex multi-hop questions that require synthesizing evidence scattered across different parts of the document. Applying a uniform strategy, such as static or manually predefined workflows, to diverse needs is inefficient; for example, complex queries often require question decomposition, whereas simple queries do not.

- Our technical contributions. To bridge this gap, we introduce BookRAG, the first retrieval-augmented generation method built upon a document-native BookIndex, designed to document

QA tasks. Specifically, to capture the deep connection of the relation in the document, BookIndex organizes information through two complementary structures. First, to preserve the document's native logical hierarchy, we organize the parsed content blocks into a hierarchical tree structure, which serves as the role of its table of contents. Second, to capture the intricate relations within these blocks, we construct a KG containing fine-grained entities. Finally, we unify these two structures by mapping the KG entities to their corresponding tree nodes.

However, effective multi-hop reasoning on the graph relies on a high-quality KG [62, 66], which is often compromised by entity ambiguity (e.g., distinct entities with names like "LLM" and "Large Language Model"). To address this, we propose a novel gradient-based entity resolution method that analyzes the similarity distribution of candidate entities. By identifying sharp drops in similarity scores, we can efficiently distinguish and merge coreferent entities, thereby ensuring graph connectivity and enhancing reasoning capabilities.

Building upon the BookIndex, we address the static of query workflows (L2) by implementing an agent-based retrieval. Specifically, our agent first classifies user queries based on their intent and complexity, and then dynamically generates tailored retrieval workflows. Grounded in Information Foraging Theory [42], our retrieval process mimics foraging by using Selector to narrow down the search space via information scents and Reasoner to locate highly relevant evidence.

We conduct extensive experiments on three widely adopted datasets to validate the effectiveness and efficiency of our BookRAG, comparing it against several state-of-the-art baselines. The experimental results demonstrate that BookRAG consistently achieves superior performance in both retrieval recall and QA accuracy across all datasets. Furthermore, our detailed analysis validates the critical contributions of our key features, such as the high-quality KG and the agent-based retrieval mechanism.

We summarize our contributions as:

- We introduce BookRAG, a novel method that constructs a document-native BookIndex by integrating a hierarchical tree of document layout blocks with a KG storing fine-grained entity relations.  
- We propose an Agent-based Retrieval approach inspired by Information Foraging Theory, which dynamically classifies queries and configures optimal retrieval workflows to locate highly relevant evidence within documents.  
- Extensive experiments on multiple benchmarks show that BookRAG significantly outperforms existing baselines, attaining state-of-the-art performance in solving complex

document QA tasks while maintaining competitive efficiency.

Outline. We review related work in Section 2. Section 3 introduces the problem formulation, IFT, and RAG workflow. In Section 4, we present the structure of our BookIndex and its construction. Section 5 presents our agent-based retrieval, elaborating on the query classification and operators used in the structured execution of BookRAG. We present the experimental results and detailed analysis in Section 6, and conclude the paper in Section 7.

# 2 RELATED WORK

In this section, we review the related works, including LLM in document analysis and the modern representative RAG approaches.

- LLM in document analysis. Recent advances in LLMs have offered opportunities to leverage LLMs in document data analysis. Due to the robust semantic reasoning capabilities of LLMs, there is an increasing number of works focusing on transferring unstructured documents (e.g., HTML, PDFs, and raw text) into structured formats, such as relational tables [1, 7, 25, 38]. For example, Evaporate [1] utilizes LLMs to synthesize extraction code, enabling cost-effective conversion of semi-structured web documents into structured databases without heavy manual annotation. In addition, several LLM-based document analysis systems have been proposed to equip standard data pipelines with semantic understanding [28, 40, 47, 53]. For instance, LOTUS [40] extends the relational model with semantic operators, allowing users to execute SQL-like queries with LLM-powered predicates (e.g., filter, join) over unstructured text corpora. Similarly, DocETL [47] introduces an agentic framework to optimize complex information extraction tasks. Furthermore, another line of research proposes to directly analyze or parse documents by viewing the document pages as images, thereby preserving critical layout and visual information [26, 31, 54].

- RAG approaches. RAG has been proven to excel in many tasks, including open-ended question answering [24, 48], programming context [9, 10], SQL rewrite [30, 50], and data cleaning [35, 36, 43]. The naive RAG technique relies on retrieving query-relevant contexts from external knowledge bases to mitigate the "hallucination" of LLMs. Recently, many RAG approaches [16, 18, 19, 21, 27, 32, 32, 45, 55, 58, 66] have adopted graph structures to organize the information and relationships within documents, achieving improved overall retrieval performance. For more details, please refer to the recent survey of graph-based RAG methods [41]. Besides, the Agentic RAG paradigm has been widely studied, employing autonomous agents to dynamically orchestrate and refine the RAG pipeline, thus significantly boosting the reasoning robustness and generation fidelity [2, 23, 59].

# 3 PRELIMINARIES

This section formalizes the research problem of complex document QA, introduces the foundational Information Foraging Theory (IFT), and briefly reviews the general workflow of RAG systems

# 3.1 Problem Formulation

We study the problem of Question Answering (QA) over complex documents, which aims to answer user queries based on long-form

documents [5, 11, 33]. Formally, a document  $D$  is represented as a sequence of  $N$  pages,  $D = \{P_{i}\}_{i = 1}^{N}$ . These pages collectively contain a sequence of content blocks  $\mathcal{B} = \{b_j\}_{j = 1}^M$ , where each block  $b_{j}$  represents a distinct element (e.g., text segment, section header, table, or image) organized within a logical chapter hierarchy. Given a user query  $q$ , the goal is to generate an accurate answer  $A$ , ideally grounded in a specific set of evidence blocks  $E\subset \mathcal{B}$ . The task is formulated as developing a method  $S$  that maps the structured document and the query to the final answer:

$$
A = \mathcal {S} (D, q) \tag {1}
$$

where  $S$  should navigate both the sequential page content and the logical hierarchy of  $D$  to synthesize the response.

# 3.2 Information Foraging Theory

Information Foraging Theory (IFT) [42] provides a framework for understanding information access as a process analogous to animal foraging. It suggests that users follow cues, known as information scent (e.g., keywords or icons), to navigate between clusters of content, known as information patches (e.g., sections in handbooks). The goal is to maximize the rate of valuable information gain while minimizing effort, guiding the decision to either stay within a patch or seek a new one.

Consider experts seeking a solution to a specific problem within a large technical handbook. They first extract key terms related to the problem, which act as information scent. This scent guides them to navigate towards one or more promising sections (the information patches). Within these patches, they analyze the diverse content to extract the precise knowledge required to formulate a final answer

# 3.3 RAG workflow

Retrieval-Augmented Generation (RAG) systems typically operate in a two-phase framework [6, 16, 41]. In the Offline Indexing phase, unstructured corpus data is organized into a structured index, which can take various forms such as vector databases or KG [66]. Subsequently, in the Online Retrieval phase, the system retrieves relevant components (e.g., text chunks or subgraphs) based on the user query  $q$  to inform the LLM's generation. However, these general workflows often treat the index as a structure derived purely from content, potentially detaching it from the document's original logical hierarchy. In contrast, our approach seeks to deeply integrate these retrieval structures with the document's native tree topology.

# 4 BOOKINDEX

This section introduces our proposed BookIndex, a hierarchical structure-aware index designed to capture both the explicit logical hierarchy and the intricate entity relations within complex documents. We first formally define the structure of the BookIndex (B). Subsequently, we elaborate on the sequential, two-stage construction process: (1) Tree Construction, which parses the document's layout to establish a hierarchical nodes, each categorized by type; and (2) Graph Construction, which extracts fine-grained entity knowledge from the tree nodes and refines it through a novel gradient-based entity resolution method.

![](images/770f339e3264523f4e165eff538a9d79948dd736587ce28100fd61d86cb6509e.jpg)

![](images/a00e76d2590b9fd8cd390a7fbc777bb0c7fed7df302318a1ba9ea8776b263563.jpg)

![](images/481e65a2b8fab730088546439a0a6b2e5297aa8fb8b3b302706209619cb741cc.jpg)  
Tree Nodes

![](images/59ec9511ea1accb73022f300d095817edd42bdcbe7a6af6d325e1b6aa7516e1c.jpg)  
GT-Link

![](images/ad594f2224f00bf859ac61ecec8d0fa0d48a55fec3e5856cf5db9cc4ca503daf.jpg)  
Entity

![](images/32802cf1519ac6fa17e3cc062dd660ce6ac23c718188bd96f39054daebbee577.jpg)  
Relation

![](images/59e31b1d4e283bf36b4432d896a8a325eaa23b585d776a387a10b79bf8a25571.jpg)  
BookIndex Construction  
Figure 2: The BookIndex Construction process. This phase includes Tree Construction, derived from Layout Parsing and Section Filtering, and Graph Construction, which involves KG Construction and Gradient-based Entity Resolution.

# 4.1 Overview of BookIndex

We formally define our BookIndex as a triplet  $B = (T, G, M)$ . Here,  $T = (N, E_T)$  represents a Tree structure where  $N$  is the set of nodes derived from the document's explicit logical hierarchy (e.g., titles, sections, tables), and  $E_T$  denotes their nesting relationships.  $G = (V, E_G)$  is a Knowledge Graph that captures fine-grained entities  $(V)$  and their relations  $(E_G)$  scattered throughout the document. Finally,  $M: V \to \mathcal{P}(N)$  is the Graph-Tree Link (GT-Link), which links each entity in  $V$  to the set of specific tree nodes in  $N$  from which it was extracted. These links are crucial for capturing the intricate, cross-sectional relations within the document. The hierarchical tree nodes in  $T$  serve as the document's native information patches, providing structured contexts for information seeking. Meanwhile, the entities and relations in  $G$ , connected via  $M$ , act as the rich information scent that guides navigation between and within these patches.

Figure 2 provides an example of our BookIndex. The Tree component, positioned at the top, organizes the document into a hierarchical structure, where content blocks such as text, tables, and images serve as leaf nodes nested within section nodes. The Graph component is composed of entities and relations extracted from these nodes. The GT-Link, illustrated by the blue dotted lines, explicitly connects these entities back to their corresponding tree nodes, thereby grounding the semantic entities within the document's logical hierarchy.

# 4.2 Tree Construction

The first stage transforms the raw document into a structured hierarchical tree  $T$ . This involves two key steps: robust layout parsing and intelligent section filtering.

4.2.1 Layout Parsing. The Layout Parsing phase processes the input document  $D$  (a collection of pages) using layout analysis and recognition models. This step identifies, extracts, and organizes diverse blocks (e.g., text, tables, images) from the document pages.

The output is a sequence of primitive blocks,  $\mathcal{B} = \{b_1, b_2, \dots, b_k\}$ , where each block  $b_i = (c_i, \tau_i, f_i)$  is defined as a triplet. Here,  $c_i$  is the raw content (e.g., text, image data),  $\tau_i$  is the initial layout-based type (e.g., Title, Text, Table, Image), and  $f_i$  is a vector of associated layout features (e.g., "FontSize", bounding box).

4.2.2 Section Filtering. Next, the Section Filtering phase processes this initial sequence to identify the document's logically hierarchical structure. Layout Parsing identifies blocks as Tit1e but does not assign their hierarchical level. Therefore, we select the candidate subset  $\mathcal{B}_{\mathrm{title}} \subset \mathcal{B}$  (where  $\tau_{i} = \mathrm{T}\mathrm{i}\mathrm{t}\mathrm{l}\mathrm{e}$ ) for an LLM-based analysis. To handle extremely long documents, this analysis is performed in batches, where each batch retains a contextual window of high-level section information (with  $l = 1$  as the root). The LLM analyzes the content  $c_{i}$  and layout features  $f_{i}$  of the candidates to determine two key properties: their actual hierarchical level  $l_{i} \in \{1,2,\ldots\}$  and final node type  $\tau_{i}'$  (e.g., re-classifying an erroneous Tit1e as Text if its level is "None"). This step is crucial for preserving the document's logical hierarchy by correcting blocks erroneously parsed as Tit1e, such as descriptive text within images or borderless table headers.

Finally, the definitive tree  $T = (N, E_T)$  is constructed. The node set  $N$  is composed of all blocks from the filtering and re-classification process, where each node  $n \in N$  retains its content  $(c_i)$  and its final node type  $(\tau_i')$  (e.g., Text, Section, Table, and Image). The edge set  $E_T$ , representing the parent-child nesting relationships, is then established. Parent-child relationships are inferred by sequentially traversing the nodes, using both the determined hierarchical levels  $(l_i)$  of Section nodes and the overall document order to assemble the complete tree structure.

As an example shown in Figure 2, the Layout Parsing phase identifies diverse blocks, typing them as Title, Text, Table, and Image. During the Section Filtering phase, the Title candidates (e.g., "Method", "Experiment", and "MOE Layer") are analyzed by the LLM. The blocks "Method" and "Experiment" (both with "FontSize: 14") are correctly identified as Section nodes at "Level: 2". Conversely,

the "MOE Layer" block ("FontSize: 20"), which was erroneously tagged as Ti tle by the parser, is re-classified by the LLM as a Text node with "Level: None". This correction is crucial for preserving the document's logical hierarchy. Following this process, all filtered and classified nodes are assembled into the final tree structure based on their determined levels and document order.

# 4.3 Graph Construction

Once the tree  $T$  is established, we proceed to populate the Knowledge Graph  $G$  by extracting and refining entities from the tree nodes.

4.3.1 KG Construction. We iterate each node  $n_i \in N$  from the previously constructed tree  $T$ . For each node  $n_i$ , we extract a subgraph  $g_i = (V_i, E_{Ri})$  based on its content  $c_i$  and final node type  $\tau_i'$ . This extraction is modality-dependent: if the node is text-only, an LLM is prompted to extract entities and relations, while for nodes containing visual elements (e.g.,  $\tau_i' = \text{Image}$ ), a Vision Language Model (VLM) is employed to extract visual knowledge. Crucially, for every entity  $v \in V_i$  extracted, its origin tree node  $n_i$  is recorded, which is vital for constructing the final mapping  $M$ .

Furthermore, to preserve structural semantics for specific logical types (e.g., Table, Formula), our process first creates a distinct, typed entity (e.g.,  $v_{\text{table}}$  representing the table itself). The other extracted entities from the specific node's content are linked to this primary vertex. For Table nodes specifically, row and column headers are also explicitly extracted as distinct entities and linked to  $v_{\text{table}}$  via a "ContainedIn" relationship.

4.3.2 Gradient-based Entity Resolution. As shown in the literature [62, 66], a well-constructed KG is essential for document question answering. A common challenge in the extraction process is that the same conceptual entity is often fragmented into multiple distinct entities due to abbreviations, co-references, or its varied occurrences across different document sections. This necessitates a robust Entity Resolution (ER) process, which identifies and merges these fragmented entities to refine the raw KG.

However, conventional ER methods are computationally expensive. They are often designed for batch processing across multiple data sources (commonly referred to as dirty ER), aiming to ensure accurate entity resolution by finding all possible matching pairs [12]. This process typically requires finding the transitive closure of all detected matches. That is, to definitively merge multiple entities (e.g., A, B, and C) as the same concept, the system must ideally compare all possible pairs ("A-B", "A-C", and "B-C") to confirm their equivalence. This can lead to a quadratic  $(O(n^{2}))$  number of pairwise comparisons, a process that becomes prohibitively slow and computationally expensive when relying on LLMs for high-accuracy judgments.

To address this, we employ a gradient-based ER method, operating on a single document (simplified as the clean ER), which performs ER incrementally as each new entity  $v_{n}$  is extracted. This transforms the quadratic batch problem into a simpler, repeated lookup task: determining where the single new entity  $v_{n}$  fits among the already-processed entities in the database. This incremental process yields two distinct, observable scoring patterns when  $v_{n}$  is reranked against its top_k most relevant candidates:

Algorithm 1: Gradient-based entity resolution  
Input:KG G,New entity  $v_{n}$  ,Rerank model R,Entity vector database DB, Vector search number top_k, threshold of gradient g // Vector Search top_k relevant entities in DB.   
1  $E_{c}\gets$  Search(DB,  $v_{n},$  top_k);   
2  $\mathcal{S}\gets \mathcal{R}(E_c,v_n)$  . // Sort all candidate entities by rerank scores.   
3 Sort(Ec,S);   
4 score  $\leftarrow S[0]$  ,Sel  $\leftarrow E_{c}[0]$  .. // Gradient select similar entities.   
5 for each remain entity  $v_{c}\in E_{c}\setminus \{E_{c}[0]\}$  do 6 if  $S[v_{c}]>$  score/g then 7 Sel  $\leftarrow$  SelU  $\{v_{c}\}$  ,score  $\leftarrow S[v_{c}]$  . 8 else break; //Merge entity or add new entity.   
9 if length(Sel)=length(Ec)then 1  $G\gets$  AddNewEntity(G,vn),DB  $\leftarrow$  AddNew(DB,vn);   
else 12 if length(Sel)=1 then  $v_{sel}\gets Sel[0]$  . 13 else  $v_{sel}\gets$  LLMSelect(Sel); 14  $G\gets$  MergeEntity(G,vn,vsel),DB  $\leftarrow$  Update(DB,vsel,vn);   
15 return G,DB;

- Case A: New Entity. If  $v_{n}$  is a new conceptual entity, its relevance scores against all existing entities will be uniformly low, showing no significant gradient or discriminative pattern.  
- Case B: Existing Entity. If  $v_{n}$  is an alias of an existing entity, its scores will show a high relevance to the true match (or a small set of equivalent aliases). Due to the reranker's inherent discriminative limitations, this initial high-relevance set might occasionally contain multiple similar entities. This high-relevance set is then typically followed by a sharp decline (a large "gradient") before transitioning to a gradual slope of irrelevant entities.

Our Gradient-based ER algorithm is designed precisely to detect this sharp decline (characteristic of Case B), allowing us to efficiently isolate the high-relevance set. Subsequently, an LLM is utilized for finer-grained distinction when multiple similar entities are identified within this set, differentiating it from the "no gradient" scenario (Case A) without quadratic comparisons.

Algorithm 1 shows the above entity resolution process. For a new entity  $v_{n}$ , we first retrieve its top_k candidates  $E_{c}$  from the vector database  $DB$ , which are then reranked by  $\mathcal{R}$  against  $v_{n}$  and sorted based on their scores  $S$  (Lines 1-3). We initialize the selection set  $Sel$  with the top-scoring candidate  $E_{c}[0]$  and set the initial score to  $S[0]$  (Line 4). We then iterate through the remaining sorted candidates (Lines 5-8). The core logic checks if the current score  $S[v_{c}]$  is still within the gradient threshold  $g$  of the previous score (i.e.,  $S[v_{c}] > \text{score} / g$ ). If the score drop is gentle (passes the check), the candidate  $v_{c}$  is added to  $Sel$ , and score is updated (Lines 7-8); otherwise, the loop breaks (Line 8) as soon as a sharp score drop is detected. Finally, the algorithm makes its decision (Lines 9-14). If the selection set  $Sel$  is identical to  $E_{c}$ , this indicates that all candidates passed the gradient check. This corresponds to Case A, where the scores lacked discriminative power (i.e.,  $v_{n}$  is equally dissimilar to

all candidates), so  $v_{n}$  is added as a new entity (Line 9-10). Conversely, if a gradient was found (i.e.,  $length(Sel) < length(E_{c})$ ), this signals Case B. We then select the canonical entity  $v_{sel}$  from Sel, using an LLM (Line 13) if the reranker identifies multiple aliases, and merge  $v_{n}$  with it (Lines 12-14). The updated  $G$  and  $DB$  are then returned (Line 15).

For instance, considering the example in Figure 2, when the new entity  $e_9$  is processed, it is first compared with existing entities in the KG. As depicted in the similarity curve (orange line),  $e_9$  shows high similarity with  $e_7$ , followed by a sharp decline in similarity with other entities like  $e_6$ ,  $e_8$ , and  $e_5$ . Our gradient-based selection process identifies  $e_7$  as the unique, high-confidence match for  $e_9$ . Consequently,  $e_9$  is merged with  $e_7$ , enriching the KG with consolidated information as shown in the final merged entity  $e_7'$ .

Graph-Tree Link (GT-Link). The GT-Link  $M$  is formalized to complete the BookIndex  $B = (T, G, M)$ . As described in the  $KG$  Construction phase, the origin tree node  $n_i$  is recorded for every newly extracted entity  $v_i$ . GT-Link is then refined during entity resolution: when an entity  $v_n$  is merged into a canonical entity  $v_{sel}$ , the origin node set of  $v_{sel}$  is updated to include all origin nodes previously associated with  $v_n$ . This aggregation process creates the final mapping  $M: V \to \mathcal{P}(N)$ , which bi-directionally links the entities in  $G$  to the set of their structural locations (nodes) in  $T$ .

# 5 AGENT-BASED RETRIEVAL

Real-world document queries are often complex, necessitating operations like modal type filtering, semantic selection, and multi-hop reasoning. To address this, we propose an agent-based approach in BookRAG, which intelligently plans and executes operations on the BookIndex. We first introduce the overall workflow and present two core mechanisms: Agent-based Planning, which formulates the strategy, and the Structured Execution, which includes the retrieval process under the principles of IFT and generation.

# 5.1 Overall Workflow

The overall workflow of agent-based retrieval, illustrated in Figure 3, follows a three-stage pipeline designed to address users' queries systematically.

1. Agent-based Planning. BookRAG first performs Classification & Plan. This stage aims to distinguish simple keyword-based queries from reasoning questions that require decomposition and analysis. For instance, a query like "How does Transformer differ from RNNs in handling long-range dependencies?" cannot be

![](images/20a0baf7e36c28e7a2358d7456a5465edaadafcca993c59eaaf194eb2e33d8e7.jpg)  
Figure 3: The general workflow of agent-based retrieval in BookRAG, which contains agent-based planning, retrieval, and generation processes.

solved by retrieving from a single keyword. Therefore, the planning stage first performs query classification. Based on this classification and a predefined set of operators designed for the BookIndex, it generates a specific operators plan that effectively guides the retrieval and generation strategies.

2. Retrieval Process. Guided by the operator plan, the retrieval process executes Scent/Filter-based Retrieval. This stage navigates the BookIndex  $B = (T, G, M)$ , either utilizing a scent-based retrieval principle (e.g., following relevant entities in  $G$ ) to find information, or employing various filters (e.g., modal type) to refine the selection. After reasoning, BookRAG gets the retrieval set of highly relevant information blocks from the BookIndex.

3. Generation Process. Finally, all retrieved information enters the generation stage for Analysis & Merging. This stage synthesizes these (often fragmented) pieces of evidence, performs final analysis, and formulates a coherent response.

# 5.2 Agent-based Planning

The planning stage is the core of BookRAG, designed to intelligently navigate our BookIndex  $B = (T, G, M)$ . To support flexible retrieval, we define four types of operators: Formulator, Selector, Reasoner, and Synthesizer. These operators can be arbitrarily combined to form tailored execution pipelines, each with adjustable parameters. BookRAG dynamically configures and assembles these operators to adapt to the specific requirements of different query categories. This process involves two sequential steps: first, the agent performs

Table 2: Three common query categories addressed in BookRAG.  

<table><tr><td>Query Category</td><td>Description</td><td>Core Task</td><td>Example Query</td></tr><tr><td>Single-hop</td><td>Queries with a single, distinct information target.</td><td>Scent-based Retrieval: Retrieve content related to a specific entity or section.</td><td>What is the definition of Information Scent?</td></tr><tr><td>Multi-hop</td><td>Queries that require synthesizing information from multiple blocks, often by decomposing into sub-problems.</td><td>Decomposing &amp; Merging: Decompose into sub-problems, retrieve for each, and synthesize the final answer.</td><td>How does Transformer differ from RNNs in handling long-range dependencies?</td></tr><tr><td>Global Aggregation</td><td>Queries that require filtering across the entire document and performing calculations.</td><td>Filter &amp; Aggregation: Apply filters across the document &amp; perform aggregation operations (e.g., Count, Sum).</td><td>How many figures related to IFT are in Section 4?</td></tr></table>

![](images/f19eaefcd84ecb53d7e7360f8f5c372378c15aba4d113c54649d4878cd00ca9b.jpg)  
Extract

![](images/eac764ddeb1ba6217e43bf1340fe79d8e42db56c5858fd09e08da858beb938ad.jpg)  
(a)Operator Set

![](images/c9f2ad7e762c9e01362fc99ef33e3b71f47bdad994f3e669c6f44696c7de0a3b.jpg)  
Entities

![](images/887d161ca8533b58e69544146a66e94302ca6dc10afccc72db982b5a3372894c.jpg)  
Decompose

![](images/120c356a6784de036cd5489e5434389d421dccca9b0f02910288fc0c5670ebb9.jpg)  
Sub-queries

![](images/60d3a482d79f153466d13fe7a58d74b076f5214dc0db5640c1d851b990ec828f.jpg)  
Formulator

![](images/93a34ab5ad5fe948339d8684152b2f4755497d92541d33eecac57b73d6f5e569.jpg)  
Filter

![](images/730d2a931fd4e294e6d3da09d88ef34eb8f3bd75a6518f82c4ed03035e940ab6.jpg)

![](images/ba5ccc0b4b44341b90d2b67e204845b76818c4c3cf35332af55a8e54946b7864.jpg)  
Operators

![](images/a39e00531416a4c1ff35758750b0cf8c9f03ae133db29c459d806da9f82b10d9.jpg)  
Select  
0-O-0  
Selector

![](images/bd8d664781f2cbd66b40b9713cd26932b666996303e90d7415895b79d901a12f.jpg)  
1  
Skyline  
S:  
.  
2

son

![](images/0793bb5febaf3afc37968f02f9959f5ad6510f80ac9faf708f28fd361f1ab273.jpg)  
Reason  
5  
06  
05  
04  
#  
.  
61

![](images/851eefb05a7d0b60f4e2f8b3699a51bfd10aa1abf4a8edb2af1a1f8d28e91e7c.jpg)  
0.6 0.  
  
5.04  
Map  
  
#  
#

![](images/4569ebd09f34d6d2ac956f104570ce2c3005507fe4b2299351e660be37b140f1.jpg)  
#  
#  
0  
A三  
#  
Reasoner

![](images/b932f3af46d22c3c9994065172ad60fde618b39621e7f5368f848e8db04bf0ff.jpg)

![](images/39c8b7a36052272e783ea8c99ef73731c71efa1d708d7cf352331ed93bf89269.jpg)

![](images/1112adbc2a0f82ec7849262d3b2a6dacf6eeb9906fe7f19ebda7d8fc3e4e79c0.jpg)

![](images/1eb4c8756e38d04a4e32f8c77eaa54ce9ffabde9ac67f5abf5892598e2bf6515.jpg)  
#  
  
#  
#  
→  
二  
#  
A=  
$\rightarrow$  
e

![](images/c9a4641fb3c324af5d749d56348cb5da0fbd43f85d3f58a456fde61dbddb33b9.jpg)  
#

![](images/06e93b3139ee5f650ce4866f2154f08611261073a16fd12988ce8576909fce63.jpg)  
+ + + + +

![](images/40257d2ef99b391ab820acc9597b9292c4c0eb91710654e7e64c5616f973a96d.jpg)  
Synthesizer

![](images/e4502da93ef401e83c3952ba7415624da847ad09f1031d8f0f9530e9b4ee6c35.jpg)

![](images/eaa3e31a2d74e725ea6251775b6494f0b35ccdc8c20340fa94a4fe2266597204.jpg)  
(b) Execution example  
Planning  
This is a Simple query...  
Operator Plan  
Extract->Select->Reason  
->Skyline->Map.  
Car and Ranking Prompt  
are entities in the  
question...

![](images/ab99814bbfbdd17a5fe1bf840430a5e6698885389014d2de98c0c04b106d08eb.jpg)  
#  
A3  
  
Prompt  
三  
等  
$\therefore m - 1 \neq  0$  ;  
  
M

![](images/07cc1f1f2117d7ae76b37ab5cc569b5825d96ed1667a7c86df3e1239f1ead412.jpg)  
$\cdots  + \overline{1} + \overline{2} + \overline{3} + \overline{4} + \overline{5} + \overline{6} + \overline{7} + \overline{8} + \overline{9} + \overline{10} + \overline{11} + \overline{12} + \overline{13} + \overline{14} + \overline{15} + \overline{16} + \overline{17} + \overline{18} + \overline{19} + \overline{20} + \overline{21} + \overline{22} + \overline{23} + \overline{24} + \overline{25} + \overline{26} + \overline{27} + \overline{28} + \overline{29} + \overline{30} + \overline{31} + \overline{32} + \overline{33} + \overline{34} + \overline{35} + \overline{36} + \overline{37} + \overline{38} + \overline{39} + \overline{40} + \overline{41} + \overline{42} + \overline{43} + \overline{44} + \overline{45} + \overline{46} + \overline{47} + \overline{48} + \overline{49} + \overline{50} + \overline{51} + \overline{52} + \overline{53} + \overline{54} + \overline{55} + \overline{56} + \overline{57} + \overline{58} + \overline{59} + \overline{60} + \overline{61} + \overline{62} + \overline{63} + \overline{64} + \overline{65} + \overline{66} + \overline{67} + \overline{68} + \overline{69} + \overline{70} + \overline{71} + \overline{72} + \overline{73} + \overline{74} + \overline{75} + \overline{76} + \overline{77} + \overline{78} + \overline{79} + \overline{80} + \overline{81} + \overline{82} + \overline{83} + \overline{84} + \overline{85} + \overline{86} + \overline{87} + \overline{88} + \overline{89} + \overline{90} + \overline{91} + \overline{92} + \overline{93} + \overline{94} + \overline{95} + \overline{96} + \overline{97} + \overline{98} + \overline{99} + \overline{{00}}.$  
Ie   
Descendant

![](images/a78d0b7e1763e437ed32adf29e06c49f437c0ee57a4049cb0407a39a9d3b9183.jpg)  
Figure 4: The BookRAG Operator Library and an Execution Example from MMLongBench dataset: (a) a visual depiction of the four operator types (Formulator, Selector, Reasoner, and Synthesizer) and (b) an execution trace for a "Single-hop" query, demonstrating the agent-based planning and step-by-step operator execution.

![](images/9b23ee16c70b63a3900e5817d0195af30b3f51c7c1fb3243ef758550ac50ef52.jpg)

![](images/2cb8a5a50b9b6d6153a9551db69023186ccaee36373e0db3d1518cbd8177088f.jpg)

![](images/3693bf2b828cdd644d4b89ff025b687d278a96a440cb565ff235c6fca552b4ae.jpg)

![](images/127a9cccb90ff1545a616329196b6f43946ee04de9bdea8a9893d6dbffe57f0a.jpg)

![](images/27798fd92477259f6ea01f2d6fc08bcd52e7376cbebe87bdc11c08cc16e8fd9e.jpg)

![](images/aaaa79654beb9b276ba18e50426590538c2b68a4accf9e9942a736f04fc3c9c9.jpg)

A: Based on the provided information the correct type of car in the Ranking Prompt Example is the Mercedes-Benz E-Class Sedan.

Query Classification to determine the appropriate solution strategy, then generates a specific Operator Plan.

- Query Classification. To enable agent strategy selection, we focus on three representative query categories defined by their intrinsic complexity and operational demands (Table 2): Single-hop, Multi-hop, and Global Aggregation. This classification is crucial because each category requires a different solution strategy. For instance, a Single-hop query typically requires a single piece of information retrieved via a Scent-based Retrieval operation. In contrast, a Global Aggregation query often necessitates analyzing content under multiple filtering conditions, usually involving a sequence of Filter & Aggregation operations across various parts of the document. Furthermore, BookRAG is designed to be extensible, allowing for the resolution of a broader range of query types by integrating additional operators.

- BookIndex Operators. To execute the strategies identified by classification, we designed a set of operators  $(O)$  tailored for the BookIndex  $B = (T,G,M)$ . These operators, visually depicted in Figure 4(a) and detailed in Table 3, define the set of operations the agent can employ for diverse query categories. We group them into four types, which we describe in sequence:

1 Formulator. These are LLM-based operators that prepare the query for execution. This category includes Decompose, which breaks a Complex query into a set of simpler, actionable sub-queries  $Q_{s}$ . It also includes Extract, which employs an LLM to identify key entities  $E_{q}$  from the query text and link them to corresponding entities in the KG,  $G$ :

$$
Q _ {s} = \operatorname {L L M} \left(P _ {\text {D e c}}, q\right) = \left\{q _ {1}, q _ {2}, \dots , q _ {k} \right\} \tag {2}
$$

$$
E _ {q} = \operatorname {L L M} \left(P _ {E x t}, q\right) = \left\{e _ {1}, e _ {2}, \dots , e _ {m} \right\} \tag {3}
$$

Here,  $q$  is the original user query, while  $P_{Dec}$  and  $P_{Ext}$  represent the prompts used to guide the LLM for the decomposition and extraction tasks, respectively.

$\Theta$  Selector. These operators filter or select specific content ranges from the BookIndex. FilterMODal and Filter_Range directly apply the explicit constraints  $C$  (e.g., modal types, page ranges) generated during the plan. Operating on the Tree  $T = (N,E_T)$ , these operators produce a filtered subset  $N_{f}$  where the predicate  $C(n)$  holds true for each node:

$$
N _ {f} = \{n \in N \mid C (n) \} \tag {4}
$$

In contrast, Select_by_Entity and Select_by_Section target contiguous document segments by retrieving subtrees rooted at specific section nodes. This process first identifies a set of target section nodes  $S_{\text{target}} \subset N$  at a specified depth, where  $S_{\text{target}}$  consists of sections either linked to entities  $E_q$  via the GT_Link  $M$  or selected by the LLM. It then retrieves all descendants of these targets to form the selected node set  $N_s$ :

$$
N _ {s} = \bigcup_ {s \in S _ {\text {t a r g e t}}} \text {S u b t r e e} (s) \tag {5}
$$

$\Theta$  Reasoner. These operators analyze and refine selected tree nodes. Graph_Reasoning performs multi-hop inference on a subgraph  $G^{\prime}(V^{\prime},E^{\prime})$  (extracted from selected nodes  $N_{s}$ ) starting from entity  $e$ . Starting from the retrieved entities, it computes an entity importance vector  $I_G \in \mathbb{R}^{|V'|}$  over the subgraph  $G^{\prime}$  using the PageRank algorithm [20]. These entity scores are then mapped to the tree nodes via the GT-Link matrix  $M$  to derive the final tree node importance scores vector  $S_G \in \mathbb{R}^{|N_s|}$ :

$$
I _ {G} = \operatorname {P a g e R a n k} \left(G ^ {\prime}, e\right) \tag {6}
$$

$$
S _ {G} = I _ {G} \times M \tag {7}
$$

Text_Ranker evaluates the semantic relevance of the tree node's content to the query  $q$ , assigning a relevance score  $S_T$  to each node. Skyline_Ranker employs the Skyline operator to filter nodes based on these multiple criteria (e.g.,  $S_G$  and  $S_T$ ), retaining only

those nodes that are not dominated by any others in terms of the specified scoring dimensions.

Synthesizer. These operators are responsible for content generation. Map performs analysis on specific retrieved information segments to generate partial responses. Reduce synthesizes a final coherent answer by aggregating information from multiple sources, such as partial answers or a collection of retrieved evidence.

- Operator Plan. After classifying the query  $(q)$  into its category  $(c)$ , the agent's final task is to generate an executable plan  $P$ . This plan is a specific sequence of operators  $\langle o_1, \ldots, o_n \rangle$  selected from our library  $O$  with parameters dynamically instantiated based on  $q$ . This process is formulated as:

$$
P = \operatorname {A g e n t} _ {\text {P l a n}} (q, c, O) \tag {8}
$$

The plan follows a structured workflow tailored to each category:

- Single-hop: The agent first attempts to Extract an entity. If successful, it executes a "scent-based" selection; otherwise, it falls back to a section-based strategy. Both paths then proceed to standard reasoning and generation, denoted as  $P_{\mathrm{std}}$ .

$$
P _ {\mathrm {s}} = \left\{\begin{array}{l}\text {E x t r a c t} \xrightarrow {\text {s u c c e s s}} \text {S e l e c t} _ {-} \text {b y} _ {-} \text {E n t i t y} \rightarrow P _ {\mathrm {s t d}}\\\text {E x t r a c t} \xrightarrow {\text {f a i l}} \text {S e l e c t} _ {-} \text {b y} _ {-} \text {S e c t i o n} \rightarrow P _ {\mathrm {s t d}}\end{array}\right. \tag {9}
$$

$$
P _ {\text {s t d}} = (\text {G r a p h} \parallel \text {T e x t}) \rightarrow \text {S k y l i n e} \rightarrow \text {R e d u c e} \tag {10}
$$

- Complex: The agent first decomposes the problem, applies the Single-hop workflow  $P_{s}$  to each sub-problem, and finally synthesizes the results.

$$
P _ {\text {c o m p l e x}} = \text {D e c o m p o s e} \rightarrow P _ {s} \rightarrow \text {M a p} \rightarrow \text {R e d u c e} \tag {11}
$$

- Global Aggregation: The workflow involves applying a sequence of filters followed by synthesis.

$$
P _ {\text {g l o b a l}} = \prod \left(\text {F i l t e r} _ {\text {M o d a l}} \mid \text {F i l t e r} _ {\text {R a n g e}}\right)\rightarrow \text {M a p} \rightarrow \text {R e d u c e} \tag {12}
$$

Here, the symbol  $\Pi$  denotes the nested composition of filters, applying either a modal or range filter at each step.

# 5.3 Structured Execution

Following the planning stage, BookRAG executes the generated workflow  $P$ . This execution phase embodies the cognitive principles of Information Foraging Theory (IFT), effectively translating abstract textual queries into concrete operations. Specifically, the Selector operators mirror the act of "navigating to information patches," narrowing the vast document space down to relevant scopes. Subsequently, the Reasoner operators perform "sensemaking within patches," where they analyze and refine the information within these focused scopes. Finally, the Synthesizer generates the answer based on the processed evidence. This design minimizes the cost of attention by ensuring computational resources are focused solely on high-value data patches.

Scent/Filter-based Retrieval. The execution begins by narrowing the scope. Aligning with IFT, Selector operators identify relevant "patches" by following "information scents" (e.g., key entities in question) or applying explicit filter constraints. This process reduces the full node set  $N$  to a focused node subset  $N_{s}$ :

$$
N _ {s} = \operatorname {S e l e c t o r} (N, \text {p a r a m s} _ {\text {s e l}}) \tag {13}
$$

This pre-selection minimizes noise and ensures that subsequent reasoning is applied only to highly relevant contexts, optimizing the foraging cost. Subsequently, within this focused scope, Reasoner operators evaluate nodes using multiple dimensions, such as graph topology and semantic relevance. We then employ the Skyline_Ranker to get the final retrieval set. Unlike fixed top- $k$  retrieval, the Skyline operator retains the Pareto frontier of nodes, retaining nodes that are valuable in at least one dimension while discarding dominated ones:

$$
N _ {R} = \text {S k y l i n e} \operatorname {R a n k e r} \left(\left\{S _ {G} (n), S _ {T} (n) \mid n \in N _ {s} \right\}\right) \tag {14}
$$

Analysis & Merging Generation. In the final stage, the Synthesizer operator generates the coherent answer by aggregating the refined evidence:

$$
A = \text {S y n t h e s i z e r} (q, N _ {R}) \tag {15}
$$

Table 3: Operators utilized in our BookRAG, categorized by their function.  

<table><tr><td>Operator</td><td>Type</td><td>Description</td><td>Parameters</td></tr><tr><td>Decompose</td><td>Formulator</td><td>Decompose a complex query into simpler, actionable sub-queries.</td><td>(Self-contained)</td></tr><tr><td>Extract</td><td>Formulator</td><td>Identify and extract key entities from the query (links to G).</td><td>(Self-contained)</td></tr><tr><td>Filter_Nodal</td><td>Selector</td><td>Filter retrieved nodes by their modal type (e.g., Table, Figure).</td><td>modal_type: str</td></tr><tr><td>Filter_Range</td><td>Selector</td><td>Filter nodes based on a specified range (e.g., pages, section).</td><td>range: (start, end)</td></tr><tr><td>Select_by_Entity</td><td>Selector</td><td>Selects all tree nodes (N) in sections linked to a given entity (V).</td><td>entity_name: str</td></tr><tr><td>Select_by_Section</td><td>Selector</td><td>Uses an LLM to select relevant sections and selects all tree nodes (N) within them.</td><td>query: str, sections: List[str]</td></tr><tr><td>Graph_Reasoning</td><td>Reasoner</td><td>Performs multi-hop reasoning on subgraph (G&#x27;) and score tree nodes (N) using graph importance and GT-links.</td><td>start-entity: str, subgraph: G&#x27;</td></tr><tr><td>Text_Reasoning</td><td>Reasoner</td><td>Rerank retrieved tree nodes (N) based on the relevance.</td><td>query: str</td></tr><tr><td>Skyline_Ranker</td><td>Reasoner</td><td>Rerank nodes based on multiple criteria.</td><td>criteria: List[str]</td></tr><tr><td>Map</td><td>Synthesizer</td><td>Uses partially retrieved information to generate a partial answer.</td><td>(Input: List[str])</td></tr><tr><td>Reduce</td><td>Synthesizer</td><td>Synthesizes the final answer from partial information or all sub-problem answers.</td><td>(Input: List[str])</td></tr></table>

The Map operator performs fine-grained analysis on individual evidence blocks or sub-problems (from Decompose) to generate intermediate insights. The Reduce operator then aggregates these partial results, such as answers to decomposed sub-queries or statistical counts from a global filter, to construct the final response. This separation ensures that the system can handle both detailed content extraction and high-level reasoning synthesis effectively.

To illustrate this end-to-end process, Figure 4(b) presents an execution trace for a "Single-hop" query: "What is the type of car in the Ranking Prompt example?" In the planning phase, the agent classifies the query and generates a specific workflow. Subsequently, it identifies key entities (e.g., "car") via Extract, retrieves relevant nodes via Select_by_Entity, refines them through reasoning and Skyline filtering, and finally synthesizes the answer using Reduce.

# 6 EXPERIMENTS

In our experiments, we evaluate BookRAG against several strong baseline methods, with an in-depth comparison of their efficiency and accuracy on document QA tasks.

# 6.1 Setup

Table 4: Datasets used in our experiments. EM and F1 denote Exact Match and F1-score, respectively.  

<table><tr><td>Dataset</td><td>MMLongBench</td><td>M3DocVQA</td><td>Qasper</td></tr><tr><td>Questions</td><td>669</td><td>633</td><td>640</td></tr><tr><td>Documents</td><td>85</td><td>500</td><td>192</td></tr><tr><td>Avg. Pages</td><td>42.16</td><td>8.52</td><td>10.95</td></tr><tr><td>Avg. Images</td><td>25.92</td><td>3.51</td><td>3.43</td></tr><tr><td>Tokens</td><td>2,816,155</td><td>3,553,774</td><td>2,265,349</td></tr><tr><td>Metrics</td><td>EM, F1</td><td>EM, F1</td><td>Accuracy, F1</td></tr></table>

Datasets & Question Synthesis. We use three widely adopted benchmarking datasets for complex document QA tasks: MMLong-Bench [33], M3DocVQA [11], and Qasper [14]. MMLongBench is a comprehensive benchmark designed to evaluate QA capabilities on long-form documents, covering diverse categories such as guidebooks, financial reports, and industry files. M3DocVQA is an open-domain benchmark designed to test RAG systems on a diverse collection of HTML-type documents sourced from Wikipedia pages<sup>1</sup>. Qasper is a QA dataset focused on scientific papers, where questions require retrieving evidence from the entire document. We filtered the datasets to remove documents with low clarity or incoherent structures. To address the scarcity of global-level questions in the original benchmarks, we synthesize additional QA pairs by having an LLM generate global questions from selected document elements (e.g., tables or figures). These questions are then answered and meticulously refined by human annotators via an outsourcing process, with this additional QA pairs constituting less than  $20\%$  of our final QA pairs. The statistics of these datasets are presented in Table 4.

Metrics. We adhere to the official metrics specified by each dataset for QA. Our primary evaluation relies on Exact Match (EM), accuracy, and token-based F1-score. To assess efficiency, we also measure time cost and token usage during the response phase. Additionally, for methods including PDF parsing, we also evaluate retrieval recall. To establish the ground truth for this, we manually label the specific PDF blocks (e.g., texts, titles, tables, images, and formulas) required to answer each question. This labeling process is guided by the metadata of ground-truth evidence provided in each dataset; we filter candidate blocks using the given modality (all datasets), page numbers (MMLongBench), and evidence statements (Qasper). Any blocks that remained non-unique after this filtering process are manually annotated. In cases where a PDF parsing error made the ground-truth item unavailable, the retrieval recall for that query is recorded as 0.

Baselines. Our experiments consider three model configurations:

- Conventional RAG: These methods are the most common pipeline for document analysis, where the raw text is first extracted and then chunked into segments of a specified size. We select strong and widely used retrieval models: BM25 [44] and Vanilla RAG. We also implement Layout+Vanilla, a variant that uses document layout analysis for semantic chunking.  
- Graph-based RAG: These methods first extract textual content from documents and then leverage graph data during retrieval. We select RAPTOR [45] and GraphRAG [16]. Specifically, GraphRAG has two versions: GraphRAG-Global and GraphRAG-Local, which employ global and local search methods, respectively.  
- Layout segmented RAG: This category encompasses methods that utilize layout analysis to segment document content into discrete structural units. We include: MM-Vanilla, which utilizes multi-modal embeddings for visual and textual content; a tree-based method inspired by PageIndex [39], denoted as TreeTraverse, where an LLM navigates the document's tree structure; DocETL [47], a declarative system for complex document processing; and GraphRanker, a graph-based method extended from HippoRAG [19] that applies Personalized PageRank [20] to rank the relevant nodes.

Implementation details. For a fair comparison, both BookRAG and all baseline methods are powered by a unified set of state-of-the-art (SOTA) and widely adopted backbone models from the Qwen family [4, 60, 63, 64]. We employ MinerU [52] for robust document layout parsing. We set the threshold of gradient  $g$  as 0.6, and more details are provided in the appendix of our technical report [57]. Our source code, prompts, and detailed configurations are available at github.com/sam234990/BookRAG.

# 6.2 Overall results

In this section, we present a comprehensive evaluation of BookRAG, analyzing its complex QA performance, retrieval effectiveness, and query efficiency compared to state-of-the-art baselines.

- QA Performance of BookRAG. We compare the QA performance of BookRAG against three categories of baselines, as shown in Table 5. The results indicate that BookRAG achieves

Table 5: Performance comparison of different methods across various datasets for solving complex document QA tasks. The best and second-best results are marked in bold and underlined, respectively.  

<table><tr><td rowspan="2">Baseline Type</td><td rowspan="2">Method</td><td colspan="2">MMLongBench</td><td colspan="2">M3DocVQA</td><td colspan="2">Qasper</td></tr><tr><td>(Exact Match)</td><td>(F1-score)</td><td>(Exact Match)</td><td>(F1-score)</td><td>(Accuracy)</td><td>(F1-score)</td></tr><tr><td rowspan="3">Conventional RAG</td><td>BM25</td><td>18.3</td><td>20.2</td><td>34.6</td><td>37.8</td><td>38.1</td><td>42.5</td></tr><tr><td>Vanilla RAG</td><td>16.5</td><td>18.0</td><td>36.5</td><td>40.2</td><td>40.6</td><td>44.4</td></tr><tr><td>Layout + Vanilla</td><td>18.1</td><td>19.8</td><td>36.9</td><td>40.2</td><td>40.7</td><td>44.6</td></tr><tr><td rowspan="3">Graph-based RAG</td><td>RAPTOR</td><td>21.3</td><td>21.8</td><td>34.3</td><td>37.3</td><td>39.4</td><td>44.1</td></tr><tr><td>GraphRAG-Local</td><td>7.7</td><td>8.5</td><td>23.7</td><td>25.6</td><td>35.9</td><td>39.2</td></tr><tr><td>GraphRAG-Global</td><td>5.3</td><td>5.6</td><td>20.2</td><td>22.0</td><td>24.0</td><td>24.1</td></tr><tr><td rowspan="4">Layout segmented RAG</td><td>MM-Vanilla</td><td>6.8</td><td>8.4</td><td>25.1</td><td>27.7</td><td>27.9</td><td>29.3</td></tr><tr><td>Tree-Traverse</td><td>12.7</td><td>14.4</td><td>33.3</td><td>36.2</td><td>27.3</td><td>32.1</td></tr><tr><td>GraphRanker</td><td>21.2</td><td>22.7</td><td>43.0</td><td>47.8</td><td>32.9</td><td>37.6</td></tr><tr><td>DocETL</td><td>27.5</td><td>28.6</td><td>40.9</td><td>43.3</td><td>42.3</td><td>50.4</td></tr><tr><td>Our proposed</td><td>BookRAG</td><td>43.8</td><td>44.9</td><td>61.0</td><td>66.2</td><td>55.2</td><td>61.1</td></tr></table>

state-of-the-art performance across all datasets, substantially outperforming the top-performing baseline by  $18.0\%$  in Exact Match on M3DocVQA. Layout + Vanilla consistently outperforms Vanilla RAG, confirming that layout parsing preserves essential structural information for better retrieval. Besides, the suboptimal results of Tree-Traverse and GraphRanker highlight the limitations of relying solely on hierarchical navigation or graph-based reasoning, which often miss cross-sectional context or drift into irrelevant scopes. In contrast, BookRAG's superiority stems from the synergy of its unified Tree-Graph BookIndex and Agent-based Planning. By effectively classifying queries and configuring optimal workflows, our BookRAG overcomes limitations of context fragmentation and static query workflow within existing baselines, ensuring precise evidence retrieval and accurate generation.

Table 6: Retrieval recall comparison among layout-based methods. The best and second-best results are marked in bold and underlined, respectively.  

<table><tr><td>Method</td><td>MMLongBench</td><td>M3DocVQA</td><td>Qasper</td></tr><tr><td>Layout + Vanilla</td><td>26.3</td><td>33.8</td><td>33.5</td></tr><tr><td>MM-Vanilla</td><td>7.5</td><td>19.7</td><td>14.9</td></tr><tr><td>Tree-Traverse</td><td>11.2</td><td>19.5</td><td>14.5</td></tr><tr><td>GraphRanker</td><td>26.4</td><td>44.5</td><td>28.6</td></tr><tr><td>BookRAG</td><td>57.6</td><td>71.2</td><td>63.5</td></tr></table>

- Retrieval performance of BookRAG. To validate our retrieval design, we evaluate the retrieval recall of BookRAG against other layout-based baselines on the ground-truth layout blocks. The experimental results demonstrate that BookRAG achieves the highest recall across all datasets, notably reaching  $71.2\%$  on M3DocVQA and significantly outperforming the next best baseline (GraphRanker, max  $44.5\%$ ). This performance advantage stems from our IFT-inspired Selector  $\rightarrow$  Reasoner workflow: the Agent-based Planning first classifies the query, enabling the Selector to narrow the search to a precise information patch, followed by the Reasoner's analysis. Crucially, after the Skyline_Ranker process, the average number of retained nodes is 9.87, 6.86, and 8.6 across the three datasets,

which is comparable to the standard top- $k$  ( $k = 10$ ) setting, ensuring high-quality retrieval without inflating the candidate size.

![](images/35980ffa566d2d8274d80ca448afe975305f73f42f888a01f4df3c35cb255981.jpg)

![](images/a1c1ac2eee2c2040646ec620823fce0c73dd5df287ef53def1ae3aa33c069bc3.jpg)

![](images/130df5aee6861a346ec369d12d72040ea48eec1590a673aebbfc1db06f0c29c0.jpg)

![](images/ce8d35b02eefeaca9d5c273bc0860a4686024e8706fde2d57ac1969854261aab.jpg)  
Figure 5: Comparison of query efficiency.

- Efficiency of BookRAG. We further evaluate the efficiency in terms of query time and token consumption, as illustrated in Figure 5. Overall, BookRAG maintains time and token costs comparable to existing Graph-based RAG methods. While purely text-based RAG approaches generally exhibit lower latency and token usage due to the absence of VLM processing for images, BookRAG maintains a balanced efficiency among multi-modal methods. In terms of token usage, BookRAG reduces consumption by an order of magnitude compared to the strongest baseline, DocETL. Notably,

on the MMLongBench dataset, DocETL consumes over 53 million tokens, whereas BookRAG requires less than 5 million. Regarding the query latency, our method also achieves a speedup of up to  $2\times$  compared to DocETL.

# 6.3 Detailed Analysis

In this section, we provide a more in-depth examination of our BookRAG. We first conduct an ablation study to validate the contribution of each component, followed by an experiment on the impact of gradient-based ER and QA performance across different query types. Furthermore, we perform a comprehensive error analysis, compare the effectiveness of our entity resolution method, and present a case study.

- Ablation study. To evaluate the contribution of each core component in BookRAG, we design several variants by removing specific components:

- w/o Gradient ER: Replaces the gradient-based entity resolution with a Basic ER by merging the same-name entities.  
- w/o Planning: Removes the Agent-based Planning, defaulting to a static, standard workflow for all queries.  
- w/o Selector: Removes the Selector operators, forcing Reasoners to score all candidate nodes.  
- w/o Graph_Reasoning: Removes the Graph_Reasoning operator. Consequently, the Skyline_Ranker is also disabled as scoring becomes single-dimensional.  
- w/o Text_Reasoning: Removes the Text_Reasoning operator. Similarly, the Skyline_Ranker is disabled, relying solely on graph-based scores.

Table 7: Comparing the QA performance of different variants of BookRAG. EM and F1 denote Exact Match and F1-score, respectively.  

<table><tr><td rowspan="2">Method variants</td><td colspan="2">MMLongBench</td><td colspan="2">Qasper</td></tr><tr><td>EM</td><td>F1</td><td>Accuracy</td><td>F1</td></tr><tr><td>BookRAG (Full)</td><td>43.8</td><td>44.9</td><td>55.2</td><td>61.1</td></tr><tr><td>w/o gradient ER</td><td>40.1</td><td>42.8</td><td>48.9</td><td>57.3</td></tr><tr><td>w/o Planning</td><td>30.8</td><td>33.2</td><td>40.9</td><td>48.5</td></tr><tr><td>w/o Selector</td><td>42.5</td><td>43.1</td><td>52.5</td><td>59.1</td></tr><tr><td>w/o Graph_Reasoning</td><td>39.8</td><td>41.5</td><td>51.4</td><td>58.4</td></tr><tr><td>w/o Text_Reasoning</td><td>39.0</td><td>40.3</td><td>47.2</td><td>52.5</td></tr></table>

The first variant evaluates the impact of KG quality on retrieval performance. The second and third variants assess the necessity of our Agent-based Planning and IFT-inspired selection mechanism, respectively. Finally, the last two variants validate the effectiveness of our multi-dimensional reasoning and dynamic Skyline filtering strategy. As shown in Table 7, the performance degradation across all variants confirms the essential role of each module in BookRAG. Specifically, the performance drop in the w/o Gradient ER variant highlights the critical role of a high-quality, connectivity-rich KG in supporting effective reasoning. Removing the Planning mechanism results in the most significant performance loss, confirming that a static workflow is insufficient for handling diverse types of queries. The w/o Selector variant, while maintaining competitive accuracy, incurs a prohibitive computational cost ( $>2 \times$  tokens on Qasper),

validating the efficiency of our IFT-inspired "narrow-then-reason" strategy.

![](images/9797ac2a8bb392c47279b8c9eb7207638b5716952db07e55452bc11f7aed36f4.jpg)  
(a) MMLongBench  
Figure 6: Comparison of graph statistics. Values are normalized to the Basic setting (Baseline=1.0). Absolute values for Basic are annotated. Note that density values are abbreviated (e.g., 3.6E-3 denotes  $3.6 \times 10^{-3}$ ).

![](images/ab24f2049d7a2d61b4728672a75b4a8bc7164dd7d5e25e24e6ee3ef51cc6e179.jpg)  
Basic Gradient-based ER  
(b) Qasper

- Impact of Gradient-based Entity Resolution. To evaluate the quality of our constructed KG, we compare the graph statistics of our Gradient-based ER against a Basic KG construction. The Basic setting employs simple exact name matching for entity merging, which is standard practice in many graph-based methods. Figure 6 presents the comparative results, normalizing the metrics (Entity count, Density, Diameter of the Largest Connected Component, and Number of Connected Components) against the Basic baseline. The results demonstrate that our Gradient-based ER significantly optimizes KG. Specifically, it reduces the number of entities (by  $12\%$ ) while substantially boosting graph density (by over  $20\%$  across datasets). This structural shift indicates that our ER module effectively identifies the same conceptual entities that possess different names. Consequently, the resulting graphs are more compact and cohesive, as evidenced by the reduced diameter and fewer connected components, which mitigates graph fragmentation and facilitates better connectivity for graph reasoning.

![](images/b450bd221315985a2e189ff5c96b9c68eca1ba637a093b41962dd431ea62df2b.jpg)  
EM / Accuracy F1-score  
(a) MMLongBench  
Figure 7: QA performance breakdown by different query types (Single-hop, Multi-hop, and Global). The blue bars represent Exact Match (EM) for MMLongBench and Accuracy for Qasper, while the red bars represent the F1-score.

![](images/b3332a59b39d31c20131ca60fd3c62b935aa2ca4fdb7e091d0c1b35d6bb91ada.jpg)  
(b) Qasper

- QA performance under different query types. Figure 7 breaks down the performance of BookRAG across Single-hop, Multihop, and Global aggregation query types. We observe that Multihop queries generally present a greater challenge compared to Single-hop ones, resulting in a slight performance decrease. This trend reflects the inherent difficulty of retrieving and reasoning over disjoint pieces of evidence. It further validates our agent-based planning strategy, which handles different query types separately.

- Error Response analysis. To diagnose the performance bottlenecks of BookRAG, we conduct a fine-grained error analysis on 200 sampled queries from each dataset, tracing the error propagation as shown in Figure 9. We categorize failures into four types:

# BookRAG response of different query types

# Single-hop Case from Qasper

Question: What is the reward model for the reinforcement learning approach?

Human-written answer: Reward 1 for successfully completing the task, with a discount by the number of turns, and reward 0 when fail.

Evidence: We defined the reward as being 1 for successfully completing the task, and 0 otherwise. A discount of  $0.95 < />$

Agent-based Planning: This is a single-hop query. Here is the Select operator:

Extract  $=$  {"entity_name": "reinforcement learning (rl)", "entity_type":"METHOD_OR_TECHNIQUE"}

Process: The Select_by_Entity operator identifies relevant sub-trees (e.g., "Introduction", "Related work"), pruning the reasoning space from 134 to 24 nodes. Subsequent Graph_Reasoning, Text_Reasoning, and Skyline_Ranker operate on this focused scope to retrieve 8 nodes for the final response.

Output: The reward model for the reinforcement learning approach is based on a binary reward system that evaluates the success or failure of dialog interactions.  $< / > 1$  for successfully completing a task.  $< / > 0$  for failing  $< / >$  using a discount factor of  $0.95 < / >$

# Multi-hop Case from Qasper

Question: What is the difference in performance between the interpretable system (e.g., vectors and cosine distance) and LSTM with ELMo system?

Human-written answer: Accuracy of best interpretable system was 0.3945 while accuracy of LSTM-ELMo net was 0.6818.

Evidence 1: The experimental results are presented in Table 1. Diacritic swapping showed a remarkably poor performance  $< / >$

Evidence 2: Test results for all the methods used. The loss measure is cross-entropy  $< / >$

Agent-based Planning: This is a multi-hop query. Here is the Decompose operator:

SubQuestion 1: "What is the performance metric of the interpretable system (e.g., vectors and cosine distance)?"

SubQuestion 2: "What is the performance metric of the LSTM with ELMo system?"

Process: BookRAG applies the single-hop workflow for each sub-question and generates the partial answers as follows:

Answer 1: "Summary:\n- Performance Metric: Accuracy (0.3945) and  $< / >$

Answer 2: "The performance metric of the LSTM with ELMo system (LSTM-ELMo net) is as follows:\nAccuracy: 0.6818 </>"

Output: The interpretable system (using vector distance and cosine distance) has an accuracy of 0.3945, while the LSTM with ELMo system has an accuracy of 0.6818. The difference in accuracy between the two systems is  $0.6818 - 0.3945 = 0.2873$ .

# Global Aggregation Case from MMLongBench

Question: How many charts are shown in the first 10 pages of the document? Human-written answer: 5

Agent-based Planning: This is a global query. Here are filter operators:

1. Filter_Range={'filter_type': page, "filter_range":"1-10"} 2. FilterMODal  $=$  {"filter_type": image}

Process: Applying filter operators retrieves images nodes from pages [3, 5, 6, 8, 9]. Map analyzes each image, and Reduce synthesizes the final output.

Output: Based on my analysis of the document, I found 5 items that answer the question.

![](images/8fe7ab97c80978d9adf559cb0f3b15e0b6a4640f5e64aa3c78f7135bd0f7384b.jpg)  
Figure 8: Case study of responses across different query types from MMLongBench and Qasper. CYAN TEXT highlights correct content generated by BookRAG. GRAY TEXT describes the internal process, and  $< / >$  marks omitted irrelevant parts.  
(a) MMLongBench  
Figure 9: Error analysis on 200 sampled queries from MM-LongBench and Qasper datasets.

![](images/a8dabc80cf30ca66eeb8ad93c3e0308e525957be15df049249a2a9cc819cb568.jpg)  
(b) Qasper

PDF Parsing, Plan, Retrieval, and Generation errors. The results identify Retrieval Error as the dominant failure mode, followed by Generation Error, reflecting the persistent challenge of locating and synthesizing multimodal evidence. Regarding Plan Error, our qualitative analysis reveals a specific failure pattern: the planner tends to over-decompose detailed single-hop queries into unnecessary multi-hop sub-tasks. This fragmentation leads to disjointed retrieval paths, effectively preventing the model from synthesizing a cohesive final answer from the scattered sub-responses.

- Case study. Figure 8 illustrates BookRAG's answering workflow across Single-hop, Multi-hop, and Global queries. The results demonstrate that by leveraging specific operators (Select, Decompose, and Filter), BookRAG effectively prunes search spaces. For example, in the Single-hop case, the reasoning space is significantly reduced from 134 to 24 nodes. This capability allows the system to efficiently isolate relevant evidence from noise, ensuring precise answer generation.

# 7 CONCLUSION

In this paper, we propose BookRAG, a novel method built upon Book Index, a document-native, structured Tree-Graph index specifically designed to capture the intricate relations of structural documents. By employing an agent-based method to dynamically configure retrieval and reasoning operators, our approach achieves state-of-the-art performance on multiple benchmarks, demonstrating significant superiority over existing baselines in both retrieval precision and answer accuracy. In the future, we will explore an integrated document-native database system that supports data formatting, knowledge extraction, and intelligent querying.

# REFERENCES

[1] Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, and Christopher Re. 2023. Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes. Proceedings of the VLDB Endowment 17, 2 (2023), 92-105.  
[2] Akari Asai, Zeqiu Wu, Yizhong Wang, et al. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In International Conference on Learning Representations (ICLR).  
[3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv preprint arXiv:2310.11511 (2023).  
[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-v1 technical report. arXiv preprint arXiv:2502.13923 (2025).  
[5] Camille Barboule, Benjamin Piwowarski, and Yoan Chabot. 2025. Survey on Question Answering over Visually Rich Documents: Methods, Challenges, and Trends. arXiv preprint arXiv:2501.02235 (2025).  
[6] Yukun Cao, Zengyi Gao, Zhiyang Li, Xike Xie, S. Kevin Zhou, and Jianliang Xu. 2025. LEGORAG: Modularizing Graph-Based Retrieval-Augmented Generation for Design Space Exploration. Proc. VLDB Endow. 18, 10 (June 2025), 3269-3283. https://doi.org/10.14778/3748191.3748194  
[7] Chengliang Chai, Jiajun Li, Yuhao Deng, Yuanhao Zhong, Ye Yuan, Guoren Wang, and Lei Cao. 2025. Doctopus: Budget-aware structural table extraction from unstructured documents. Proceedings of the VLDB Endowment 18, 11 (2025), 3695-3707.  
[8] Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: The muppets straight out of law school. arXiv preprint arXiv:2010.02559 (2020).  
[9] Sibei Chen, Yeye He, Weiwei Cui, Ju Fan, Song Ge, Haidong Zhang, Dongmei Zhang, and Surajit Chaudhuri. 2024. Auto-Formula: Recommend Formulas in Spreadsheets using Contrastive Learning for Table Representations. Proceedings of the ACM on Management of Data 2, 3 (2024), 1-27.  
[10] Sibei Chen, Nan Tang, Ju Fan, Xuemi Yan, Chengliang Chai, Guoliang Li, and Xiaoyong Du. 2023. Haipipe: Combining human-generated and machine-generated pipelines for data preparation. Proceedings of the ACM on Management of Data 1, 1 (2023), 1-26.  
[11] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multi-modal retrieval is a key for multi-page multi-document understanding. arXiv preprint arXiv:2411.04952 (2024).  
[12] Vassilis Christophides, Vasilis Efthymiou, Themis Palpanas, George Papadakis, and Kostas Stefanidis. 2020. An overview of end-to-end entity resolution for big data. ACM Computing Surveys (CSUR) 53, 6 (2020), 1-42.  
[13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025).  
[14] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. arXiv preprint arXiv:2105.03011 (2021).  
[15] Xavier Daull, Patrice Bellot, Emmanuel Bruno, Vincent Martin, and Elisabeth Murisasco. 2023. Complex QA and language models hybrid architectures, Survey. arXiv preprint arXiv:2302.09051 (2023).  
[16] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024).  
[17] Yunfàn Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 (2023).  
[18] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv e-prints (2024), arXiv-2410.  
[19] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. arXiv preprint arXiv:2405.14831 (2024).  
[20] Taher H Haveliwala. 2002. Topic-sensitive pagerank. In Proceedings of the 11th international conference on World Wide Web. 517-526.  
[21] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. arXiv preprint arXiv:2402.07630 (2024).  
[22] Yucheng Hu and Yuxing Lu. 2024. Rag and rau: A survey on retrieval-augmented language model in natural language processing. arXiv preprint arXiv:2404.19543 (2024).  
[23] Soyeong Jeong, Jinheon Baek, et al. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. arXiv preprint arXiv:2403.14403 (2024).

[24] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. 2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. arXiv preprint arXiv:2403.14403 (2024).  
[25] Tengjun Jin, Yuxuan Zhu, and Daniel Kang. 2025. ELT-Bench: An End-to-End Benchmark for Evaluating AI Agents on ELT Pipelines. arXiv preprint arXiv:2504.04808 (2025).  
[26] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangwoo Yun, Dongyoon Han, and Seunghyun Park. 2022. Ocr-free document understanding transformer. In European Conference on Computer Vision. Springer, 498-517.  
[27] Dawei Li, Shu Yang, Zhen Tan, Jae Young Baik, Sukwon Yun, Joseph Lee, Aaron Chacko, Bojan Hou, Duy Duong-Tran, Ying Ding, et al. 2024. DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature. arXiv preprint arXiv:2405.04819 (2024).  
[28] Guoliang Li, Jiayi Wang, Chenyang Zhang, and Jiannan Wang. 2025. Data+ AI: LLM4Data and Data4LLM. In Companion of the 2025 International Conference on Management of Data. 837-843.  
[29] Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen. 2023. Large language models in finance: A survey. In Proceedings of the fourth ACM international conference on AI in finance. 374-382.  
[30] Zhaodonghui Li, Haitao Yuan, Huiming Wang, Gao Cong, and Lidong Bing. 2025. LLM-R2: A Large Language Model Enhanced Rule-based Rewrite System for Boosting Query Efficiency. Proceedings of the VLDB Endowment 1, 18 (2025), 53-65.  
[31] Haoyu Lu, Wen Liu, Bo Zhang, et al. 2024. DeepSeek-VL: Towards Real-World Vision-Language Understanding. arXiv preprint arXiv:2403.05525 (2024).  
[32] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. 2024. Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation. arXiv preprint arXiv:2407.10805 (2024).  
[33] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems 37 (2024), 95963-96010.  
[34] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511 (2022).  
[35] Zan Ahmad Naeem, Mohammad Shahmeer Ahmad, Mohamed Eltabakh, Mourad Ouzzani, and Nan Tang. 2024. RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes. Proceedings of the VLDB Endowment 17, 12 (2024), 4421-4424.  
[36] Avanika Narayan, Ines Chami, Laurel Orr, and Christopher Re. 2022. Can Foundation Models Wrangle Your Data? Proceedings of the VLDB Endowment 16, 4 (2022), 738-746.  
[37] Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M Mulvey, H Vincent Poor, Qing-song Wen, and Stefan Zohren. 2024. A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges. arXiv preprint arXiv:2406.11903 (2024).  
[38] Arash Dargahi Nobari and Davood Rafiei. 2024. TabulaX: Leveraging Large Language Models for Multi-Class Table Transformations. arXiv preprint arXiv:2411.17110 (2024).  
[39] PageIndex. 2025. PageIndex: Next-Generation Reasoning-based RAG. https://pageindex.ai/.  
[40] Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos Guestrin, and Matei Zaharia. 2025. Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS. Proceedings of the VLDB Endowment 18, 11 (2025), 4171-4184.  
[41] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohoe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey. arXiv preprint arXiv:2408.08921 (2024).  
[42] Peter Pirolli and Stuart Card. 1995. Information foraging in information access environments. In Proceedings of the SIGCHI conference on Human factors in computing systems. 51-58.  
[43] Yichen Qian, Yongyi He, Rong Zhu, Jintao Huang, Zhijian Ma, Haibin Wang, Yaohua Wang, Xiuyu Sun, Defu Lian, Bolin Ding, et al. 2024. UniDM: A Unified Framework for Data Manipulation with Large Language Models. Proceedings of Machine Learning and Systems 6 (2024), 465-482.  
[44] Stephen E Robertson and Steve Walker. 1994. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In SIGIR'94: Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University. Springer, 232-241.  
[45] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. 2024. Raptor: Recursive abstractive processing for tree-organized retrieval. arXiv preprint arXiv:2401.18059 (2024).  
[46] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2024.

Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems 36 (2024).  
[47] Shreya Shankar, Tristan Chambers, Tarak Shah, Aditya G Parameswaran, and Eugene Wu. 2024. Docet!: Agentic query rewriting and evaluation for complex document processing. arXiv preprint arXiv:2410.12189 (2024).  
[48] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kalurachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. Transactions of the Association for Computational Linguistics 11 (2023), 1-17.  
[49] Solutions Review Editors. 2019. 80 Percent of Your Data Will Be Unstructured in Five Years. https://solutionsreview.com/data-management/80-percent-of-your-datawill-be-unstructured-in-five-years/. Accessed: 2023-10-27.  
[50] Zhaoyan Sun, Xuanhe Zhou, and Guoliang Li. 2024. R-Bot: An LLM-based Query Rewrite System. arXiv preprint arXiv:2412.01661 (2024).  
[51] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports 9, 1 (2019), 1-12.  
[52] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. 2024. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839 (2024).  
[53] Jiayi Wang and Guoliang Li. 2025. Aop: Automated and interactive llm pipeline orchestration for answering complex queries. CIDR.  
[54] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024).  
[55] Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. 2025. ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation. arXiv preprint arXiv:2502.09891 (2025).  
[56] Shen Wang, Tianlong Xu, Hang Li, Chaoli Zhang, Joleen Liang, Jiliang Tang, Philip S Yu, and Qingsong Wen. 2024. Large language models for education: A survey and outlook. arXiv preprint arXiv:2403.18105 (2024).

[57] Shu Wang, Yingli Zhou, and Yixiang Fang. [n.d.]. BookRAG: A Hierarchical Structure-aware Index-based Approach for Complex Document Question Answering. https://github.com/sam234990/BookRAG.  
[58] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. 2024. Knowledge graph prompting for multi-document question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 19206-19214.  
[59] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval Augmented Generation. arXiv preprint arXiv:2401.15884 (2024).  
[60] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).  
[61] Murong Yue. 2025. A survey of large language model agents for question answering. arXiv preprint arXiv:2503.19213 (2025).  
[62] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Junnan Dong, et al. 2025. A survey of graph retrieval-augmented generation for customized large language models. arXiv preprint arXiv:2501.13958 (2025).  
[63] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. GME: Improving Universal Multimodal Retrieval by Multimodal LLMs. arXiv preprint arXiv:2412.16855 (2024).  
[64] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176 (2025).  
[65] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 1, 2 (2023).  
[66] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, et al. 2025. In-depth Analysis of Graph-based RAG in a Unified Framework. arXiv preprint arXiv:2503.04338 (2025).  
[67] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. 2023. Large language models for information retrieval: A survey. ACM Transactions on Information Systems (2023).

# A EXPERIMENTAL DETAILS

# A.1 Evaluation Metrics

In this section, we provide the detailed definitions and calculation procedures for the metrics used in our main experiments.

A.1.1 Answer Extraction and Normalization. Standard RAG models typically generate free-form natural language responses, which may contain extraneous conversational text (e.g., "The answer is..."). Directly comparing these raw outputs with concise ground truth labels (e.g., "Option A" or "12.5") can lead to false negatives.

Following official evaluation protocols, we employ an LLM-based extraction step to align the model output with the ground truth format before calculation. Let  $y_{raw}$  denote the raw response generated by the RAG system and  $y_{gold}$  denote the ground truth. We define the extracted answer  $\hat{y}$  as:

$$
\hat {y} = \mathrm {L L M} _ {\text {e x t r a c t}} \left(y _ {r a w}, \text {I n s t r u c t i o n}\right) \tag {16}
$$

where  $\mathrm{LLM}_{\mathrm{extract}}$  extracts the key information (e.g., the key entity for span extraction) from  $y_{\mathrm{raw}}$ . We further apply standard normalization  $\mathcal{N}(\cdot)$  (e.g., lowercasing, removing punctuation) to both  $\hat{y}$  and  $y_{\mathrm{gold}}$ .

A.1.2 QA Performance Metrics. Based on the ground truth  $y_{gold}$  and the model's response (either raw  $y_{raw}$  or extracted  $\hat{y}$ ), we compute the following metrics:

Accuracy (Inclusion-based). Following prior works [3, 34, 46], we utilize accuracy as a soft-match metric. We consider a prediction correct if the normalized gold answer is included in the model's generated response, rather than requiring a strict exact match. This accounts for the uncontrollable nature of LLM generation.

$$
\text {A c c u r a c y} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} \left(\mathcal {N} \left(y _ {\text {g o l d}, i}\right) \subseteq \mathcal {N} \left(y _ {\text {r a w}, i}\right)\right) \tag {17}
$$

where  $\subseteq$  denotes the substring inclusion relation.

Exact Match (EM). Unlike accuracy, Exact Match is a strict metric. It measures whether the normalized extracted answer  $\hat{y}$  is character-for-character identical to the ground truth.

$$
\operatorname {E M} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} \left(\mathcal {N} \left(\hat {y} _ {i}\right) = \mathcal {N} \left(y _ {\text {g o l d}, i}\right)\right) \tag {18}
$$

F1-score. For questions requiring text span answers, we utilize the token-level F1-score between the extracted answer  $\hat{y}$  and the ground truth  $y_{gold}$ . Treating them as bags of tokens  $T_{\hat{y}}$  and  $T_{gold}$ :

$$
P = \frac {\left| T _ {\hat {y}} \cap T _ {g o l d} \right|}{\left| T _ {\hat {y}} \right|}, \quad R = \frac {\left| T _ {\hat {y}} \cap T _ {g o l d} \right|}{\left| T _ {g o l d} \right|}, \quad F 1 = \frac {2 \cdot P \cdot R}{P + R} \tag {19}
$$

A.1.3 Retrieval Recall. As described in the main text, we evaluate retrieval quality based on the granularity of parsed PDF blocks (e.g., paragraphs, tables, images). For a given query  $q$ , let  $\mathcal{B}_{gold}$  be the set of manually labeled ground-truth blocks required to answer  $q$ , and  $\mathcal{B}_{ret}$  be the set of unique blocks retrieved by the system. The Retrieval Recall is defined as:

$$
\operatorname {R e c a l l} _ {r e t} = \left\{ \begin{array}{l l} 0 & \text {i f p a r s i n g e r r o r o c c u r s o n} \mathcal {B} _ {\text {g o l d}} \\ \frac {\left| \mathcal {B} _ {\text {r e t}} \cap \mathcal {B} _ {\text {g o l d}} \right|}{\left| \mathcal {B} _ {\text {g o l d}} \right|} & \text {o t h e r w i s e} \end{array} \right. \tag {20}
$$

Specifically, if a ground-truth block is lost due to PDF parsing failures (i.e., it does not exist in the candidate pool), it is considered strictly unretrievable, resulting in a recall contribution of 0 for that specific block.

# A.2 Implementation details

We implement BookRAG in Python, utilizing MinerU [52] for robust document layout parsing. For a fair comparison, both BookRAG and all baseline methods are powered by a unified set of state-of-the-art (SOTA) and widely adopted backbone models from the Qwen family [4, 60, 63, 64], including LLM, vision-language model (VLM), and embedding models. Specifically, we utilize Qwen3-8B [60] as the default LLM, Qwen2.5VL-30B [4] as the vision-language model (VLM), Qwen3-Embedding-0.6B [64] for text embedding, gme-Qwen2-VL-2B-Instruct [63] for multi-modal embedding, and Qwen3-Reranker-4B [64] for reranking. We primarily select models under the 10B parameter scale to balance efficiency and effectiveness. However, for the VLM, we adopt the 30B version, as the 8B counterpart exhibited significant performance deficits, frequently failing to answer correctly even when provided with ground-truth images. All experiments were conducted on a Linux operating system running on a high-performance server equipped with an Intel Xeon 2.0GHz CPU, 1024GB of memory, and 8 NVIDIA GeForce RTX A5000 GPUs, each with 24 GB of VRAM. Specifically, to ensure a fair comparison of efficiency, all methods were executed serially, and the reported time costs reflect this sequential processing mode. For methods involving document chunking and retrieval ranking, we standardize the chunk size at 500 tokens and set the retrieval top-k to 10 to ensure consistent candidate pool sizes across baselines. For further reproducibility, our source code and detailed implementation configurations are publicly available at our repository: https://github.com/sam234990/BookRAG.

# A.3 Prompts

Specifically, we present the prompts designed for agent-based query classification (Figure 10), question decomposition (Figure 11), and filter operator generation (Figure 12). Additionally, we illustrate the prompt employed for entity resolution judgment (Figure 13) during the graph construction phase.

You are an expert query analyzer. Your only task is to classify the user's question into one of three categories: "simple", "complex", or "global". Respond only with the specified JSON object.

# Category Definitions:

1. single-hop: The question can be fully answered by retrieving information from a SINGLE, contiguous location in the document (e.g., one specific paragraph, one complete table, or one figure).

- This includes questions that require reasoning or comparison, as long as all the necessary data is present within that single retrieved location.  
- Example: "What is the title of Figure 2?"  
- Example: "How do  $5\%$  of the Latinos see economic upward mobility for their children?" -> This is SIMPLE because the answer can be found by looking at a single chart or paragraph.

2. multi-hop: The question requires decomposition into multiple simple sub-questions, where each sub-question must be answered by a separate retrieval action.

- It often contains a nested or indirect constraint that requires a preliminary step to resolve before the main question can be answered.  
- Example: "What is the color of the personality vector...?" -> This is COMPLEX because it requires two separate retrieval actions.

3. global: The question requires an aggregation operation (e.g., counting, listing, summarizing) over a set of items that are identified by a clear structural filter.

- Example: "How many tables are in the document?" -> This is GLOBAL because the process is to filter for all items of type 'table'.

User Query: query

Figure 10: The prompt for query classification.

```txt
You are a query decomposition expert. You have been given a "complex" question. Your task is to break it down into a series of simple, atomic sub-questions and classify each one by type.  
**Crucial Instructions:**  
1. Each `retrieval` sub-question MUST be a direct information retrieval task that can be answered independently by looking up a specific fact, number, or value in the document.  
2. **retrieval` sub-questions MUST NOT depend on the answer of another sub-question.** They should be parallelizable. All logic for combining their results must be placed in a final `synthesis` question.  
3. A `synthesis` question requires comparing, calculating, or combining the answers of the previous `retrieval` questions. It does **NOT** require a new lookup in the document.  
You MUST provide your response in a JSON object with a single key 'sub_questions', which contains a list of objects. Each object must have a 'question' (string) and a 'type' (string: "retrieval" or "synthesis").  
--- EXAMPLE 1 (Correct Decomposition with Independent Lookups) ---  
Complex Query: "What is the color of the personality vector in the soft-labeled personality embedding matrix that with the highest Receptiviti score for User A2GBIFL43U1LKJ?"  
Expected JSON Output:  
{{"sub_questions": [{"question": "What are all the Receptiviti scores for each personality vector for User A2GBIFL43U1LKJ?", "type": "retrieval"}], {"question": "What is the mapping of personality vectors to their colors in the soft-labeled personality embedding matrix?", "type": "retrieval"}}, {"question": "From the gathered scores, identify the personality vector with the highest score, and then find its corresponding color from the vector-to-color mapping.", "type": "synthesis"}]  
}}  
--- END EXAMPLE 1 ---  
--- EXAMPLE 2 (Decomposition with retrieval and synthesis steps) ---  
Complex Query: "According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?"  
Expected JSON Output:  
{{"sub_questions": [{"question": "According to the report, what is the population of foreign born Latinos in the survey?", "type": "retrieval"}], {"question": "According to the report, what is the population of Latinos interviewed by cellphone in the survey?", "type": "retrieval"}], {"question": "Which of the two population counts is greater?", "type": "synthesis"}]  
}}  
--- END EXAMPLE 2 ---  
Now, perform the decomposition for the following query.  
User Query: query
```

Figure 11: The prompt for query decomposition.

```txt
You are a highly specialized AI assistant. Your only function is to analyze a "Global Query" and return a single, valid JSON object that specifies both the filtering steps and the final aggregation operation. You MUST NOT output any other text or explanation.  
```bash
##INSTRUCTIONS \& DEFINITIONS  
1. **Filters**: You MUST determine the list of `filters` to apply. Even if the filter is for the whole document (e.g., all tables), the `filters` list must be present.  
- `filter_type': One of ["section", "image", "table", "page"].  
- `section': Use for structural parts like chapters, sections, appendices, or references.  
- `image': Use for visual elements like figures, images, pictures, or plots.  
- `table': Use for tabular data.  
- `page': Use for specific page numbers or ranges.  
- `filter_value': (Optional) Can be provided for "section" (e.g., a section title) or "page" (e.g., '3-10' or '5').  
**For "image" or "table", this value MUST be null.**  
2. **Operation**: Determine the final aggregation operation.  
- `operation': One of ["COUNT", "LIST", "SUMMARIZE", "ANALYZE"]  
## EXAMPLES OF YOUR TASK  
User: "How many figures are in this paper from Page 3 to Page 10?"  
Assistant: {{"filters": {{"filter_type": "page", "filter_value": "3-10"}}}, {{"filter_type": "image"}}], "operation": "COUNT"]}  
User: "Summarize the discussion about 'data augmentation' in the 'Methodology' section."  
Assistant: {{"filters": {{"filter_type": "section", "filter_value": "Methodology"}}], "operation": "SUMMARIZE"]}  
User: "How many chapters are in this report?"  
Assistant: {{"filters": {{"filter_type": "section"}}], "operation": "COUNT"]}  
##YOUR CURRENT TASK  
User: {"query"}  
User Query: query
```

Figure 12: The prompt for Filter operator generation.

```markdown
-Goal-
You are an expert Entity Resolution Adjudicator. Your task is to determine if a "New Entity" refers to the exact same real-world concept as one of the "Candidate Entities" provided from a knowledge graph. Your output must be a JSON object containing the ID of the matching candidate (or -1) and a brief explanation for your decision.
-Context-
You will be given one "New Entity" recently extracted from a text. You will also be given a list of "Candidate Entities" that are semantically similar, retrieved from an existing knowledge base. Each candidate has a unique `id' for you to reference.
---Core Task & Rules-
1. **Analyze the "New Entity"*: Carefully read its name, type, and description to understand what it is.
2. **Field-by-Field Adjudication**: To determine a match, you must evaluate each field with a specific focus:
* **entity_name* (High Importance): ** The names must be extremely similar, a direct abbreviation (e.g., "LLM" vs. "Large Language Model"), or a well-known alias. **If the names represent distinct, parallel concepts (like "Event Detection" and "Named Entity Recognition"), they are NOT a match, even if their descriptions are very similar.
* **entity_type* (Medium Importance): ** The types do not need to be identical, but they must be closely related and compatible (e.g., 'COMPANY' and 'ORGANIZATION' could describe the same entity).
* **description* (Contextual Importance): ** The descriptions may differ as they are often extracted from different parts of a document. Your task is to look past surface-level text similarity and determine if they fundamentally describe the **same underlying object or concept**.
3. **Be Strict and Conservative**: Your standard for a match must be very high. An incorrect merge can corrupt the knowledge graph. A missed merge is less harmful.
* Surface-level similarities are not enough. The underlying concepts must be identical.
* For example, "Apple" (the fruit) and "Apple Inc." (the company) are NOT a match.
* **When in doubt, you MUST output -1**
* **Assume No Match by Default**: In a large knowledge graph, most new entities are genuinely new. You should start with the assumption that the "New Entity" is unique. You must find **strong, convincing evidence** across all fields, especially the 'entity_name', to overturn this assumption and declare a match.
4. **Format the Output**: **You must provide your answer in a valid JSON format. The JSON object should contain two keys:** *`select_id': An integer. The `id' of the candidate you've determined to be an exact match. If no exact match is found, this value MUST be `-1'. *`explanation': A brief, one-sentence string explaining your reasoning. For a match, explain why they are the same entity. For no match, explain the key difference.
---Output Schema & Format-
Your response MUST be a single, valid JSON object that adheres to the following schema. Do not include any other text, explanation, or markdown formatting like ***json.
***json
{{ "select_id": "integer", "explanation": "string"
}}
---***
-Example-
### Example 1: Match Found
### Example 2: No Match Found
----Task Execution-
Now, perform the selection task based on the following data. Remember to output only a single integer.
- Input Data -
```

Figure 13: The prompt for entity resolution judgement, examples are omitted due to lack of space.