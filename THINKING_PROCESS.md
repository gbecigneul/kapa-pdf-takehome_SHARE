# Thinking process

- After conducting research, it seems that Docling would be a better candidate for PDF -> Markdown conversion than PyMu. 
- I edited the project so that we can use Docling and run those two side by side for comparison of:
    - PDF -> Markdown conversion
    - Chunking
    - Chat app end-to-end usage
- In this process I made sure we:
    - can use MLX/MPS acceleration on Apple M chips if available, as Docling supports it out of the box with automatic device detection. Have a look at my screenshot `assets/asitop_apple_gpu_usage.png` that confirms M-chip GPU usage during Docling indexing
    - use different tables when indexing documents with the two separate converters to avoid data conflicts, by slightly editing the in-memory vector store and where it's passed
- While using the app I noticed that 
    - (+) Docling beats PyMu at nicely formatting sentences that are on a page transition (`ex1_*` attached screenshots, see `assets/`)
    - (-) Docling sometimes creates too much formatting, i.e. inventing tables or section-subsection hierarchy where they don't exist (`ex2_*` attached screenshots, see `assets/`)
- As a result, Docling's Markdown results in chunks that are too small. When using the RAG app, this causes the app to retrieve too small chunks with not enough information. When asking "What courses should I take to become a judge for criminal cases", the app with PyMu converter provides a better answer
- I then ran some investigations as to what updates could be made to my Docling integration to circumvent these issues. I identified potential candidate updates such as:
    - Trying to increase the chunk size of the chunker
    - Post-processing the Docling generated Markdown to flatten overly deep headings or remove empty headings
    - Looking at the 'Advanced options' that Docling exposes: https://docling-project.github.io/docling/usage/advanced_options/ and I found:
        - Try `TableFormerMode.ACCURATE` versus `TableFormerMode.FAST` for better table detection
            - I tried this and it didn't resolve the table hallucinations I found. However going forward I keep the "accurate" mode, as conversion is not a latency critical job.
        - Log the .md that Docling produces so that I can read the .md file in a code editor to understand what happens at the heading level (see `data/md_logs` dir I added)
            - I tried this and understood that Docling outputs everything at '##' heading level, which is actually a known issue! https://github.com/docling-project/docling/issues/1023 -> that means any flattening or post/reformatting I tried of the Docling converting will be useless because Docling kills the hierarchical structure already... 

- This realization suggests that Docling is not that mature actually, despite being released by IBM and its 37k github stars...
- After more research, it seems we have other good candidates: 
    - Marker (28k stars): https://github.com/datalab-to/marker
    - Markitdown (72k stars, by Microsoft): https://github.com/microsoft/markitdown
    - Unstructured (12k stars): https://github.com/Unstructured-IO/unstructured
    - Nougat (9k stars, by Meta, good for academic papers): https://github.com/facebookresearch/nougat

- Out of curiosity I tried increasing the chunk sizes significantly (which is not allowed by the instructions) and I noticed that it significantly improved the capacity of even pymu to answer the challenge questions of the REAMDE. But since it's not the focus of the task, I won't explore this, and I'm back on improving the converter only.

- I implemented converter files for Markitdown and Marker, and adapted the dashboard to now show comparison between the 4 converters we have: pymu, docling, markitdown, marker. 
- Marker takes way too long on my hardware (Macbook with Chip M3 Pro) timing out after 20min. It achieves 99% GPU utilisation (see asitop screenshot in `assets/`) which is good, but it takes too long, probably because of vision caps trying to OCR all images. Lets just ignore that one for now. 
- Markitdown Runs fast and is able to properly answer the challenging question "Which department offers the course Quantitative Methods?" however it fails at many other questions. Inspecting its chunks I see that headings are all removed which can explain why it fails many questions. After investigating it seems this is a known issue: https://github.com/microsoft/markitdown/issues/304! Again! How can these tools be so popular and fail at the most basic elements of their core purpose...

See `assets/comparison.png` to see what the new dashboard looks like, comparing converters side by side. 

- Next try: Unstructured: https://github.com/Unstructured-IO/unstructured
    - turns out the chunks are way too small. That's caused by it using huge headings everywhere. 
    - I tried to log the raw json and raw text (see `data/md_logs/*.raw.json`) produced by its extractor before I convert it to a .md, to see if the issue was with the final step. But no, even in the Json, "Title" fields all have the same "layout_width" and "layout_height" and no other param or other json field would allow to differentiate them from each other...

- Next: I saw pymumdf actually has parameters that can be tuned e.g. the way headers are handled. That's the method I called pymu_hdr. Unfortunately it would require more time investment to potentially yield better results.

Since I have spent slightly over a day, I am handing over the assignment in its current state.

ENDING 

Please see in `assets/` for final comparisons on the sample questions the files `final_comparison_{i}.png`.


The subsequent sections (staging and prod) would only be relevant thoughts after better performance on retrieval would have been achieved.






# Staging

If spending more time improving quality of the pipeline, I would mainly:
- run more extensive research on all the possible existing methods for pdf to md
- especially try models incorporating vision capabilities, e.g. OCR and caption generation of images. I would potentially use dedicated models for tables understanding
- I would investigate leveraging multimodal capabilities of existing main providers (OpenAI API, and others) as they are already capable of processing PDFs
- I would add custom prefixes [prefix] to sections of the mardown so that I can re-manipulate its structure post conversion, to make it more performant on the many test cases that I would manually collect the document conversions. 

# Production

In prod envs I would consider:
- Making this sync Python pipeline fully async with `asyncio`, e.g. as a FastAPI app. Use `uvloop` as default event loop policy which is a good prod practice. 
- Using Docling's inference server to execute the model inferences instead of doing them in-process (if using Docling at all)
- Offloading CPU-intensive work to micro-services which could run in their own containers on a CPU serverless infra (e.g. AWS ECS Fargate) with horizontal auto-scaling (default policy on AWS Fargate is scaling at >75% CPU usage, that would fit). So that our FastAPI app remains 100% I/O bound. Using external micro-services in that case is safer at scale than using asyncio.to_thread() or even than offloading to an external process pool executor. I have experience with all of those scenari. 
- Profiling to investigate if some optimizations would make sense such as potentially:
    - Reducing memory footprint of some of our class objects using the `__slots__` attribute.
    - Cythonizing / Numba-converting some big loops
    - If we need to cache some class objects to run things faster, use `@dataclass(frozen=True)` to make them hashable.
- Make sure we use cache where possible, e.g. Redis.
- Design unit tests based on the edge/error cases I discovered, so that for each update I make, I can easily know if it improves/worsens wrt those, for light-speed iterations. I can go beyong Pytest and implement my own LLM-as-a-judge tests, for those that are "soft". 
- If very high load, investigate whether we could use a Triton inference server (grpc, configurable dynamic batching) or even export the models PyTorch->ONNX->TensorRT if we want to go one-step further. Even speed is even more critical, we could also use Groq GPUs for inference instead of Nvidia. However in practice I would assume convertion and chunking can be mostly precomputed, and hence not super latency-critical.

