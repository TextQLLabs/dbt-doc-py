from typing import Optional, Dict, List, Tuple
import inquirer
import os
import sys
import json
from enum import Enum
import yaml
from ruamel.yaml import YAML
from dataclasses import dataclass
import itertools
import threading
from transformers import GPT2Tokenizer, AutoTokenizer
import asyncio
import argparse
import httpx
import subprocess

stdout_lock = threading.Lock()

class OAIRequest:
    def __init__(self, model: str, prompt: str, temperature: float, max_tokens: int):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

class OAIRequestWithUserInfo:
    def __init__(self, prompt: str, email: str):
        self.prompt = prompt
        self.email = email

class OAIChoice:
    def __init__(self, text: str):
        self.text = text

class OAIResponse:
    def __init__(self, choices: List[OAIChoice]):
        self.choices = choices

class ColumnMetadata:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class ArguParseException(Exception):
    def __init__(self, message):
        super().__init__(message)

class Depends:
    def __init__(self, nodes: Optional[List[str]] = None, macros: Optional[List[str]] = None):
        self.nodes = nodes
        self.macros = macros

class NodeMetadata:
    def __init__(
        self,
        original_file_path: str,
        patch_path: Optional[str] = None,
        compiled_code: Optional[str] = None,
        raw_code: Optional[str] = None,
        description: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        resource_type: Optional[str] = None,
        package_name: Optional[str] = None,
        path: Optional[str] = None,
        alias: Optional[str] = None,
        checksum: Optional[str] = None,
        config: Optional[str] = None,
        tags: Optional[str] = None,
        meta: Optional[str] = None,
        group: Optional[str] = None,
        docs: Optional[str] = None,
        build_path: Optional[str] = None,
        deferred: Optional[str] = None,
        unrendered_config: Optional[str] = None,
        created_at: Optional[str] = None,
        name: str = None,
        unique_id: str = None,
        fqn: List[str] = None,
        columns: Dict[str, ColumnMetadata] = None,
        depends_on: Optional[Depends] = None,
    ):
        self.original_file_path = original_file_path
        self.patch_path = patch_path
        self.compiled_code = compiled_code
        self.raw_code = raw_code
        self.description = description
        self.database = database
        self.schema = schema
        self.resource_type = resource_type
        self.package_name = package_name
        self.path = path
        self.alias = alias
        self.checksum = checksum
        self.config = config
        self.tags = tags
        self.meta = meta
        self.group = group
        self.docs = docs
        self.build_path = build_path
        self.deferred = deferred
        self.unrendered_config = unrendered_config
        self.created_at = created_at
        self.name = name
        self.unique_id = unique_id
        self.fqn = fqn
        self.columns = columns
        self.depends_on = depends_on

class Manifest:
    def __init__(self, nodes: Dict[str, NodeMetadata]):
        self.nodes = nodes

class KeyOrUserInfo:
    def __init__(self, key: Optional[str] = None, user_info: Optional[str] = None):
        self.key = key
        self.user_info = user_info

class Env:
    def __init__(
        self,
        api_key: KeyOrUserInfo,
        base_path: str,
        project_name: str,
        models: Optional[set[str]] = None,
        dry_run: bool = False,
    ):
        self.api_key = api_key
        self.base_path = base_path
        self.project_name = project_name
        self.models = models
        self.dry_run = dry_run

@dataclass
class Arguments:
    pass

@dataclass
class Working_Directory(Arguments):
    path: str

class Gen_Undocumented(Arguments):
    pass

@dataclass
class Gen_Specific(Arguments):
    models_list: str

class DbtDocGen(Arguments):
    pass

class Dry_Run(Arguments):
    pass

class GenMode(Enum):
    undocumented = 1
    specific = 2

documented_nodes = {}

class ArgsConfig:
    def __init__(self, working_directory: str, gen_mode: GenMode, dbtDocGen: bool, dry_run: bool):
        self.working_directory = working_directory
        self.gen_mode = gen_mode
        self.dbtDocGen = dbtDocGen
        self.dry_run = dry_run

def mk_prompt(reverse_deps: Dict[str, List[str]], node: NodeMetadata) -> str:
    deps = ",".join(node.depends_on.nodes) if node.depends_on and node.depends_on.nodes else "(No dependencies)"
    r_deps = (
        ",".join(reverse_deps[node.unique_id])
        if node.unique_id in reverse_deps
        else "Not used by any other models"
    )
    staging = "\nThis is a staging model. Be sure to mention that in the summary.\n" if "staging" in node.fqn else ""
    raw_code = node.raw_code if node.raw_code else ""

    prompt = f"""Write markdown documentation to explain the following DBT model. Be clear and informative, but also accurate. The only information available is the metadata below.
    Explain the raw SQL, then explain the dependencies. Do not list the SQL code or column names themselves; an explanation is sufficient.

    Model name: {node.name}
    Raw SQL code: {raw_code}
    Depends on: {deps}
    Depended on by: {r_deps}
    {staging}
    First, generate a human-readable name for the table as the title (i.e. fct_orders -> # Orders Fact Table).
    Then, describe the dependencies (both model dependencies and the warehouse tables used by the SQL.) Do this under ## Dependencies.
    Then, describe what other models reference this model in ## How it's used
    Then summarize the model logic in ## Summary.
    """
    return prompt

def mk_column_prompt(node: NodeMetadata, col: ColumnMetadata, documented_nodes: Dict[str, NodeMetadata]) -> str:
    # Check if the current column depends on any documented nodes
    deps = node.depends_on.nodes if node.depends_on and node.depends_on.nodes else []
    inherited_docs = []
    for dep in deps:
        if dep in documented_nodes:
            # Add the name and description of each column in the dependent node to the list of inherited docs
            for dep_col in documented_nodes[dep].columns.values():
                if dep_col.name in dep_col.depends_on.columns:
                    inherited_docs.append(dep_col.description)

    # Combine the inherited docs into a single string
    inherited_docs_str = "\n\n".join(inherited_docs)
    inheritance = f"""This column is inherited from another model. Use this column's documentation from the original model 
    as context for writing the requested one and be sure to mention it alongside the name of the original model. 
    Inherited documentation: {inherited_docs_str} """ if inherited_docs_str else ""

    prompt = f"""Write markdown documentation to explain the following DBT column in the context of the parent model and SQL code. Be clear and informative, but also accurate. The only information available is the metadata below.
    Do not list the SQL code or column names themselves; an explanation is sufficient.

    Column Name: {col.name}
    Parent Model name: {node.name}
    Raw SQL code: {node.raw_code}
    {inheritance}

    First, explain the meaning of the column in plain, non-technical English. Then, explain how the column is extracted in code.
    If the column is calculated from other columns, explain how the calculation works.
    If the column is derived from other columns, explain how those columns are extracted.
    If the column is a inherited from another model, mention the original model and use the provided Inherited documentation (If there is any).
    Remember not to list the SQL code, brackets, or the word config (Jinja code); an explanation is sufficient.
    """
    return prompt

class SummarizedResult:
    def __init__(
        self,
        patch_path: Optional[str],
        summary: str,
        original_file_path: str,
        column_summaries: Dict[str, str],
        name: str,
    ):
        self.patch_path = patch_path
        self.summary = summary
        self.original_file_path = original_file_path
        self.column_summaries = column_summaries
        self.name = name

class TooManyTokensError(Exception):
    pass

async def run_openai_request(env: Env, prompt: str) -> str:
    resGPT=""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")        
        #tokens = tokenizer.encode(prompt)        
        tokens = tokenizer(prompt, truncation=True, max_length=2000)        

        if len(tokens) + 1000 >= 4096:
            raise TooManyTokensError()

        temp = 0.2

        base_req = OAIRequest(
            model="text-davinci-003", prompt=prompt, temperature=temp, max_tokens=1000            
        )
        
        if env.api_key.key:
            url = "https://api.openai.com/v1/completions"            
            headers = {
                "Authorization": f"Bearer {env.api_key.key}",
                "Content-Type": "application/json",
            }
            data = json.dumps(base_req.__dict__)
        else:
            url = "https://api.textql.com/api/oai"
            headers = {"Content-Type": "application/json"}
            body = OAIRequestWithUserInfo(prompt=prompt, email=env.api_key.user_info)
            data = json.dumps(body.__dict__)
            print(f"Using TextQL API with user info {env.api_key.user_info}")
            print(f"Data {data}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, data=data, timeout=60)        
        
        #print(str(response.json()))
        if 'error' in response.json():
            print(response.json()['error']['message'])            
        else:
            result = OAIResponse(choices=[OAIChoice(text=c["text"]) for c in response.json()["choices"]])
            resGPT = result.choices[0].text

    except Exception as e:
        print(e)
        #raise e    

    return resGPT
    #return "respuesta GPT"

async def gen_column_summaries(env: Env, node: NodeMetadata) -> Dict[str, str]:
    prefix = "[ai-gen] "
    
    async def mapper(k: str, column: ColumnMetadata) -> Tuple[str, str]:
        result = await run_openai_request(env, mk_column_prompt(node, column, documented_nodes))    
        return (k, prefix + result)
    
    filtered_columns = {k: v for k, v in node.columns.items() if v.description == ""}
    
    result_seq = await asyncio.gather(*(mapper(k, col) for k, col in filtered_columns.items()))

    return dict(result_seq)    

async def open_ai_summarize(env: Env, reverse_deps: Dict[str, List[str]], node: NodeMetadata) -> Optional[SummarizedResult]:
    with stdout_lock:
        print(f"Generating docs for: {node.name}")

    summary_prefix = "This description is generated by an AI model. Take it with a grain of salt!\n"

    try:
        tbl_result, col_result = await asyncio.gather(
            run_openai_request(env, mk_prompt(reverse_deps, node)),
            gen_column_summaries(env, node)
        )
        return SummarizedResult(
            patch_path=node.patch_path,
            name=node.name,
            original_file_path=node.original_file_path,
            summary=summary_prefix + tbl_result,
            column_summaries=col_result
        )
    except TooManyTokensError:
        with stdout_lock:
            print(f"Prompt for {node.name} returned too many tokens to fit into GPT-3. Perhaps the SQL code or dependency map is too large?")
        return None
    except Exception as e:
        with stdout_lock:
            print(f"OAI request to {node.name} failed: {e}")
        raise e  # Reraise the exception to trigger a retry

def insert_column_description(env, node_result: SummarizedResult, col_map: Dict[str, str], model_node) -> None:
    model_node_ = model_node

    name_node = model_node_["name"]

    name = name_node

    if name not in col_map:
        return

    col_result = col_map[name]
    doc_name = f"tql_generated_doc__{node_result.name}__{name}"

    md_path = os.path.join(
        env.base_path,
        os.path.dirname(node_result.original_file_path),
        f"{doc_name}.md"
    )

    header = f"{{% docs {doc_name} %}}"
    footer = "{% enddocs %}"

    doc_content = "\n".join([header, col_result, footer])

    with stdout_lock:
        print(f"Writing new docs to: {md_path}")

    if env.dry_run:
        print(doc_content)
    else:
        with open(md_path, "w") as f:
            f.write(doc_content)

    ## Removing the `columns` from dictionary only to be added after `description`
    model_node_.pop("description", None)
    columns = model_node_.pop("columns", None)
    model_node_["description"] = f"{{{{ doc(\"{doc_name}\") }}}}"
    model_node_["columns"] = columns

def insert_description(env, node_map: Dict[str, SummarizedResult], model_node) -> None:    
    model_node_ = model_node

    name_node = model_node_["name"]

    name = name_node

    if name not in node_map:        
        return
        
    node = node_map[name]
    doc_name = f"tql_generated_doc__{node.name}"

    md_path = os.path.join(
        env.base_path,
        os.path.dirname(node.original_file_path),
        f"{doc_name}.md"
        )

    header = f"{{% docs {doc_name} %}}"
    footer = "{% enddocs %}"

    doc_content = "\n".join([header, node.summary, footer])

    with stdout_lock:
        print(f"Writing new docs to: {md_path}")

    if "columns" in model_node_:
        cols_node = model_node_["columns"]
    
        for col in cols_node:
            insert_column_description(env, node, node.column_summaries, col)

    if env.dry_run:
        print(doc_content)
    else:
        with open(md_path, "w") as f:
            f.write(doc_content)

    model_node_.pop("description", None)
    model_node_["description"] = f"{{{{ doc(\"{doc_name}\") }}}}"
        
def insert_docs(env: Env, patch_path_may: Optional[str], nodes: List[SummarizedResult]) -> None:
    if patch_path_may is None:        
        return
    
    path = os.path.join(env.base_path, patch_path_may.replace(f"{env.project_name}://", ""))

    with open(path, "r") as f:
        contents = f.read()

    deserializer = yaml.SafeLoader(contents)
    config = deserializer.get_single_data()

    data = yaml.safe_load(contents)
    models = data["models"]

    result_map = {n.name: n for n in nodes}

    models_node = config["models"]

    for model_obj in models_node:
        model_name = model_obj['name']
        model = model_obj.get('columns', [])  # If 'columns' doesn't exist, default to an empty list.        
        if model_name in result_map:
            insert_description(env, result_map, model_obj)

    with stdout_lock:
        print(f"Adding description to {len(nodes)} models in {path}")

    if env.dry_run:
        ## Prints to console
        YAML().dump(config, sys.stdout)
    else:
        ## Writes to yaml file
        with open(path, "w") as f:
            YAML().dump(config, f)

def read_project_config(base_path: str) -> str:
    path = os.path.join(base_path, "dbt_project.yml")
    
    with open(path, "r") as f:
        contents = f.read()

    deserializer = yaml.SafeLoader(contents)
    config = deserializer.get_single_data()

    data = yaml.safe_load(contents)
    name_node = data["name"]
    
    return name_node

def is_model(name: str) -> bool:
    node_type = name.split('.')[0]
    return node_type == "model"

def should_write_doc(env: Env, pair: Tuple[str, NodeMetadata]) -> bool:
    def pred(nm):
        if env.models is None:
            return pair[1].description == ""
        return nm in env.models

    has_patch_path = pair[1].patch_path is not None

    cond = is_model(pair[0]) and pred(pair[1].name)

    if not has_patch_path and cond:
        print(f"Model {pair[0]} doesn't appear to be declared in a .yml file. Generating docs isn't yet supported for models without a corresponding yaml declaration.")
        user_input = input("Do you want to generate the missing .yml file? Y/n: ")
        if user_input.lower() == 'y':
            generateYaml(env,pair[1])
            return True

    return has_patch_path and cond

def generateYaml(env: Env,node_metadata: NodeMetadata):
    # Get the path to the .yml file
    yaml_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), node_metadata.name + '.yml')

    try:
        catalog_path = os.path.join(env.base_path, "target", "catalog.json")
        catalog = getCatalog(catalog_path)
    except Exception as e:
        print("catalog.json deserialization failed")
        raise e

    # Extract column names
    column_list = []
    for model_name, (columns, types) in catalog.items():
        if model_name == node_metadata.name:
            column_list = [{'name': col_name} for col_name in columns]

    # Create the data structure for the YAML file
    data = {
        "version": 2,
        "models": [
            {
                "name": node_metadata.name,
                "columns": column_list
            }
        ]
    }

    # Write the data structure to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print(f"YAML file successfully generated at {yaml_file_path}!")

def mk_reverse_dependency_map(nodes: Dict[str, NodeMetadata]) -> Dict[str, List[str]]:
    ans: Dict[str, List[str]] = {}

    def folder(nm: str, metadata: NodeMetadata) -> None:
        nodes = metadata.depends_on.nodes if metadata.depends_on else []

        if is_model(nm):
            for model_dep in nodes:
                if model_dep in ans:
                    ans[model_dep].append(nm)
                else:
                    ans[model_dep] = [nm]

    for key, value in nodes.items():
        folder(key, value)

    return ans

class ApiKeyNotFound(Exception):
    pass

def parse_args(argv) -> ArgsConfig:
    parser = argparse.ArgumentParser(prog="DbtHelper")
    
    parser.add_argument("--working-directory", type=str, default="./", help="Specify the working directory.")
    
    gen_mode_group = parser.add_mutually_exclusive_group()
    gen_mode_group.add_argument("--undocumented", dest="gen_mode", action="store_const", const=GenMode.undocumented, default=GenMode.undocumented, help="Use undocumented gen mode.")
    gen_mode_group.add_argument("--specific", dest="gen_mode", type=lambda s: GenMode.specific(list(s.split(','))), help="Use specific gen mode with a list of models.")
    
    parser.add_argument("--dbtDocGen", type=bool, default=True, help="Run command `dbt docs generate` automatically.")

    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode.")
    
    args = parser.parse_args(argv)

    return ArgsConfig(working_directory=args.working_directory, gen_mode=args.gen_mode, dbtDocGen=args.dbtDocGen, dry_run=args.dry_run)

def parse_columns(json_data: Dict[str, dict]) -> Dict[str, ColumnMetadata]:
    columns = {}
    for column_name, column_data in json_data.items():
        if 'depends_on' in column_data:
            column_data['depends_on'] = Depends(**column_data['depends_on'])

        # Extract only the fields needed for ColumnMetadata
        column_data_subset = {
            k: column_data.get(k) for k in
            ['name', 'description']
        }

        columns[column_name] = ColumnMetadata(**column_data_subset)
    return columns

def parse_node_metadata(json_data: Dict[str, dict]) -> Dict[str, NodeMetadata]:
    node_metadata = {}
    for node_id, node_data in json_data.items():
        columns = parse_columns(node_data.get('columns', {}))
        node_data['columns'] = columns
        if 'depends_on' in node_data:
            node_data['depends_on'] = Depends(**node_data['depends_on'])

        # Extract only the fields needed for NodeMetadata
        node_data_subset = {
            k: node_data.get(k) for k in
            ['original_file_path', 'patch_path', 'compiled_code', 'raw_code', 'description',
             'database', 'schema', 'resource_type', 'package_name', 'path', 'alias', 'checksum',
             'config', 'tags', 'meta', 'group', 'docs', 'build_path', 'deferred', 'unrendered_config',
             'created_at', 'name', 'unique_id', 'fqn', 'columns', 'depends_on']
        }

        node_metadata[node_id] = NodeMetadata(**node_data_subset)
    return node_metadata

def parse_manifest(json_data: Dict[str, dict]) -> Manifest:
    node_metadata = parse_node_metadata(json_data['nodes'])
    return Manifest(nodes=node_metadata)

def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_manifest_from_json(file_path: str) -> Manifest:
    json_data = read_json_file(file_path)
    return parse_manifest(json_data)

def getCatalog(json_file: str) -> Dict[str, List[str]]:
    with open(json_file) as f:
        data = json.load(f)
    
    models_columns = dict()
    
    # Loop through nodes, there might be multiple models
    for node in data['nodes'].values():
        # Get model name
        model_name = node['metadata']['name']
        
        # Get column names
        columns = [column_info['name'] for column_info in node['columns'].values()]
        
        # Get column types
        types = [column_info['type'] for column_info in node['columns'].values()]
        
        # Add model and its columns to the result
        models_columns[model_name] = (columns, types)
    
    return models_columns

def get_nodes_with_description(manifest_path: str) -> Dict[str, NodeMetadata]:
    manifest = load_manifest_from_json(manifest_path)
    nodes_with_description = {}
    for node_name, node_metadata in manifest.nodes.items():
        if node_metadata.description != "":
            nodes_with_description[node_name] = node_metadata
    return nodes_with_description

class UserInfo:
    def __init__(self, email: str):
        self.email = email

class Key:
    def __init__(self, key: str):
        self.key = key

def run_dbt_docs_generate(path_to_dbt_project, arg_dbtDocGen):
    if arg_dbtDocGen:
        try:
            print("Running DBT docs generate...")
            subprocess.check_output(['dbt', 'docs', 'generate'], cwd=path_to_dbt_project)
            print("DBT docs successfully generated!")
        except subprocess.CalledProcessError as e:
            print("Could not generate DBT docs.")
            print("Error:")
            print(e.output)
    else:
        print("DBT docs generation is turned off... Make sure to run `dbt docs generate` before and after running this tool.")

### Metrics ###
async def generateMetrics(env: Env, selected_nodes: Dict[str, NodeMetadata]):
    for node_name, node_metadata in selected_nodes:
        print("Generating metrics for model: " + node_name)        

        # Get the path to the .yml file
        #yaml_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), "tql_generated_metric_" + node_metadata.name + '_')

        try:
            catalog_path = os.path.join(env.base_path, "target", "catalog.json")
            catalog = getCatalog(catalog_path)
        except Exception as e:
            print("catalog.json deserialization failed")
            raise e

        # Extract column names and types for the model
        column_list = []
        for model_name, (column_names, column_types) in catalog.items():
            if model_name == node_metadata.name:
                column_list = [{col_name : col_type} for col_name, col_type in zip(column_names, column_types)]

        #Asking the user for a base requirement for the metric
        promptDes = promptDesignMetrics(node_metadata, column_list)
        #print("promptDes: " + promptDes)
        res1 = await run_openai_request(env, promptDes)
        #print('Res GPT design: ' + res1)
        queries = [block.strip() for block in res1.split(";")]
        #print('List of queries: ' + str(queries))
        
        # Create the data structure for the YAML file
        for rawquery in queries:
            if len(rawquery.split(":")) == 2:
                metric_name, query = rawquery.split(":")
                prompt = promptGenMetrics(node_metadata, column_list, query, metric_name)

                print("Generating metric: " + metric_name + "\n" + "With the prompt: " + query)

                #Call OpenAI API
                result = await run_openai_request(env, prompt)
                result = result.replace("\\n", "").replace("\\\\", "")

                # Write the data structure to the YAML file
                yaml_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), "tql_genmetric_" + metric_name + '.yml')            
                with open(yaml_file_path, 'w') as yaml_file:
                    yaml_file.write(result)                
                    print(f"Metrics YAML file successfully generated at {yaml_file_path}!")

                sqlPrompt = promptGenMetricSQL(metric_name, result)
                sqlText = await run_openai_request(env, sqlPrompt)

                # Write the query in to a SQL file
                sql_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), "metric_" + metric_name + '.sql')            
                with open(sql_file_path, 'w') as sql_file:
                    sql_file.write(sqlText)
                    print(f"Metrics SQL file successfully generated at {sql_file_path}!")

            else:
                continue        

def promptDesignMetrics(node: NodeMetadata, column_list: List[str]):
    prompt = f"""You are a data analyst that receives the structure of a DBT model/table and outputs statements for the generation of DBT metrics and dimension for the model/table.
    You are given the metadata of a DBT model and a list of columns and data types in that model.
    You return three statements separated by this character: ';'
    Remember, just 3 statements! and the final one should not have the character: ';' at the end.
    Each statement is a prompt for the generation of a DBT metric and/or dimension that will be done by another LLM. You also should return the name of the metric that you intend to create with the prompt.
    You should separate the name of the metric from the prompt with this character: ':'. Remember, the name of the metric should always be at the left side of the ':', and the prompt at the rigt one.
    The generated metrics should be diverse and helpful for the analysis of the model/table.
    These are some guidelines for the generation of metrics:
    {metricsGuidelines}

    These are some examples of generation of statements for the generation of metrics:

    Generic Example 1:

    Input:
    model: dim_customers
    model_description: A table containing all customers
    sql_query: SELECT * FROM dim_customers
    columns: ['customer_id', 'customer_name', 'customer_email', 'customer_phone', 'customer_address', 'customer_city', 'customer_state', 'customer_country', 'lifetime_value', 'is_paying', 'company_name', 'signup_date', 'plan']
    
    Output:
    new_customers:new customers using the product by month and half of the month;
    total_revenue:total revenue (lifetime_value) by the following dimensions customer_name, customer_country, and company_name;
    customers_count:number of customers by the following dimensions customer_city, customer_country, and company_name

    Generic Example 2:

    Input:
    model: order_events
    model_description: model used by the finance team to better understand revenue but inconsistencies in how it's reported have led to requests that the data team centralize the definition in the dbt repo.
    sql_query: SELECT * FROM order_events
    columns: ['event_date', 'order_id', 'order_country', 'order_status', 'customer_id', 'customer_status', 'amount']
    
    Output:
    total_revenue:Total revenue by year and country;
    total_orders:Total orders by customer status and country;
    average_revenue:Average revenue by year and country

    Complete the following. Return the three statements separated by ; with no ther comment.
    model: {node.name}
    model_description: {node.description}
    sql_query: {node.raw_code}
    columns: {column_list}    
    Output:
    """
    return prompt

def promptGenMetrics(node: NodeMetadata, column_list: List[str], query: str, metricName: str):
    prompt = f"""You are an expert SQL analyst with a large knowledge of the DBT platform that takes natural language input and outputs DBT metrics in YAML format.
    You are given the metadata of a DBT model, the list of columns and data types in that model, and the name of the metric.
    
    follow the following guidelines to generate the metric:
    {metricsGuidelines}

    Some examples of metric generation are:

    Generic Example 1:

    model: dim_customers
    model_description: A table containing all customers
    sql_query: SELECT * FROM dim_customers
    columns: ['customer_id', 'customer_name', 'customer_email', 'customer_phone', 'customer_address', 'customer_city', 'customer_state', 'customer_country', 'lifetime_value', 'is_paying', 'company_name', 'signup_date', 'plan']
    metric_name: new_customers
    Input: new customers using the product in half of the month.
    YAML Output:version: 2

metrics:
  - name: new_customers
    label: New Customers
    model: ref('dim_customers')
    description: "The 14 day rolling count of paying customers using the product"

    calculation_method: count_distinct
    expression: customer_id 

    dimensions:
      - plan
      - customer_country

    filters:
      - field: is_paying
        operator: 'is'
        value: 'true'
      - field: lifetime_value
        operator: '>='
        value: '100'
      - field: company_name
        operator: '!='
        value: "'Acme, Inc'"
      - field: signup_date
        operator: '>='
        value: "'2020-01-01'"


    Generic Example 2:

    model: dim_customers
    model_description: A table containing all customers
    sql_query: SELECT * FROM dim_customers
    columns: ['customer_id', 'customer_name', 'customer_email', 'customer_phone', 'customer_address', 'customer_city', 'customer_state', 'customer_country', 'lifetime_value', 'is_paying', 'company_name', 'signup_date', 'plan']
    metric_name: average_revenue_per_customer
    Input: average revenue per customer, segment data per plan and country.
    YAML Output:version: 2

metrics:
  - name: average_revenue_per_customer
    label: Average Revenue Per Customer
    model: ref('dim_customers')
    description: "The average revenue received per customer"

    calculation_method: average
    expression: lifetime_value 

    dimensions:
      - plan
      - customer_country
        

    Complete the following. Use YAML format and include no other commentary.
    model: {node.name}
    columns: {column_list}
    model_description: {node.description}
    sql_query: {node.raw_code}
    metric_name: {metricName}
    Input: {query}
    YAML Output:"""
    return prompt

def promptGenMetricSQL(metric: str, metricName: str):
    prompt = f"""You are an expert SQL analyst with a large knowledge of the DBT platform that takes DBT metrics in YAML format as input and outputs queries in SQL format.
    You are given the definition and the name of the metric.

    Generic Example 1:
    
    metric_name: new_customers    
    metric:version: 2

metrics:
  - name: new_customers
    label: New Customers
    model: ref('dim_customers')
    description: "The 14 day rolling count of paying customers using the product"

    calculation_method: count_distinct
    expression: customer_id 

    dimensions:
      - plan
      - customer_country

    filters:
      - field: is_paying
        operator: 'is'
        value: 'true'
      - field: lifetime_value
        operator: '>='
        value: '100'
      - field: company_name
        operator: '!='
        value: "'Acme, Inc'"
      - field: signup_date
        operator: '>='
        value: "'2020-01-01'"

    SQL Output:select * 
from {{{{ metrics.calculate(
    metric('new_customers'),    
    dimensions=['plan', 'customer_country']    
) }}}}


    Generic Example 2:
    
    metric_name: average_revenue_per_customer    
    metric:version: 2

metrics:
  - name: average_revenue_per_customer
    label: Average Revenue Per Customer
    model: ref('dim_customers')
    description: "The average revenue received per customer"

    calculation_method: average
    expression: lifetime_value 

    dimensions:
      - plan
      - customer_country
        
    SQL Output:select * 
from {{{{ metrics.calculate(
    metric('average_revenue_per_customer'),    
    dimensions=['plan', 'customer_country']    
) }}}}

    Complete the following. Use SQL format and include no other commentary.
    Keep in mind that between the from clause and the metrics.calculate() clause, 
    you should use two brackets for opening and two for closing (For instance: from <double brackets> metrics.calculate(parameters) <double brackets>).
    metric_name: {metricName}
    metric: {metric}
    SQL Output:"""
    return prompt

metricsGuidelines = """General Information:
    A metric is an aggregation over a table that supports zero or more dimensions.
    Metric names must:
        contain only letters, numbers, and underscores (no spaces or special characters)
        begin with a letter
        contain no more than 250 characters


    Available properties (Define aspects of the metric)
    Field: name
    Description: A unique identifier for the metric
    Example: new_customers
    ---
    Field: model
    Description: The dbt model that powers this metric
    Example: dim_customers
    ---
    Field: label
    Description: A short for name / label for the metric
    Example: New Customers
    ---
    Field: description
    Description: Long form, human-readable description for the metric
    Example: The number of customers who....
    ---
    Field: calculation_method
    Description: The method of calculation (aggregation or derived) that is applied to the expression
    Example: count_distinct
    ---
    Field: expression
    Description: The expression to aggregate/calculate over
    Example: user_id, cast(user_id as int)
    Guidelines: You can't use * as expression in the metric. It can be used only as a SQL expression (for instance count(*))
                For expressions that needs math operations (sum,average,median) make sure you are using a numeric column (acoording the provided data type of the column).
    ---
    Field: timestamp
    Description: 
    Example: signup_date
    ---
    Field: time_grains
    Description: One or more "grains" at which the metric can be evaluated. For more information, see the "Custom Calendar" section.
    Example: [day, week, month, quarter, year]
    ---
    Field: dimensions
    Description: A list of dimensions to group or filter the metric by
    Example: [plan, country]
    ---
    Field: window
    Description: A dictionary for aggregating over a window of time. Used for rolling metrics such as 14 day rolling average. Acceptable periods are: [day,week,month, year, all_time]
    Example: {{count: 14, period: day}}
    ---
    Field: filters
    Description: A list of filters to apply before calculating the metric
    Guideline: Filters should be defined as a list of dictionaries that define predicates for the metric. Filters are combined using AND clauses. For more control, users can (and should) include the complex logic in the model powering the metric.
            All three properties (field, operator, value) are required for each defined filter.
    ---
    


    Available calculation methods (The method of calculation (aggregation or derived) that is applied to the expression)
    Method: count
    Description: This metric type will apply the count aggregation to the specified field
    ---
    Method: count_distinct
    Description: This metric type will apply the count aggregation to the specified field, with an additional distinct statement inside the aggregation
    ---
    Method: sum
    Description: This metric type will apply the sum aggregation to the specified field
    ---
    Method: average
    Description: This metric type will apply the average aggregation to the specified field
    ---
    Method: min
    Description: This metric type will apply the min aggregation to the specified field
    ---
    Method: max
    Description: This metric type will apply the max aggregation to the specified field
    ---
    Method: median
    Description: This metric type will apply the median aggregation to the specified field, or an alternative percentile_cont aggregation if median is not available
    ---
    Method: derived
    Description: This metric type is defined as any non-aggregating calculation of 1 or more metrics
    ---
    """
### Metrics ###

async def main(argv) -> int:    
    try:
        args_env = parse_args(sys.argv[1:])

        run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)

        manifest_path = os.path.join(args_env.working_directory, "target", "manifest.json")        

        try:
            manifest = load_manifest_from_json(manifest_path)
        except Exception as e:
            print("manifest.json deserialization failed")
            raise e        

        try:            
            project_name = read_project_config(args_env.working_directory)
        except Exception as e:
            print("Reading dbt_project.yml failed. Please re-run from a dbt project root.")
            raise e

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            print("You haven't specified an API Key. No worries, this one's on TextQL!")
            print("In return, please type your email address. We don't collect any other data, nor sell your email to third parties.")
            print("If you're okay with this, press enter. Otherwise, type 'no' and set the OPENAI_API_KEY environment variable.")
            email = input("Email (type no to abort): ")

            if email == "no":
                raise ApiKeyNotFound()
            else:
                user_info = UserInfo(email)
        else:
            user_info = Key(api_key)

        models = None if args_env.gen_mode == GenMode.undocumented else set(args_env.gen_mode.value)

        documented_nodes = get_nodes_with_description(manifest_path)

        init = (manifest, Env(api_key=user_info, base_path=args_env.working_directory, project_name=project_name, models=models, dry_run=args_env.dry_run))

    except ArguParseException as e:
        print(e.message)
        return 1
    except Exception as e:
        print("Initialization failed. Aborting")
        print(e)
        return 1

    manifest, env = init

    if env.dry_run:
        print("Dry Run. Results will not be written.")
    
    r_deps = mk_reverse_dependency_map(manifest.nodes)

    ### Model Selection ###
    options = [(k, (k, v)) for k, v in manifest.nodes.items()]

    questions = [
        inquirer.Checkbox('models',
                        message='Select the models you want to document: (Press Spacebar to select, Enter to confirm) ',
                        choices=options,
                        ),
    ]

    answers = inquirer.prompt(questions)

    selected_nodes = dict(answers['models']).items()
    ### Model Selection ###

    nodes_to_process = [pair for pair in selected_nodes if should_write_doc(env, pair)]

    #Assigning generated .yml files to nodes
    for node in nodes_to_process:
        if node[1].patch_path is None:
            node[1].patch_path = os.path.join(env.base_path, os.path.dirname(node[1].original_file_path), node[1].name + ".yml")
    
    summarized_nodes = await asyncio.gather(*[open_ai_summarize(env, r_deps, pair[1]) for pair in nodes_to_process])
    summarized_nodes = [node for node in summarized_nodes if node is not None]
    
    for patch_path, group in itertools.groupby(sorted(summarized_nodes, key=lambda x: x.patch_path), key=lambda x: x.patch_path):
        insert_docs(env, patch_path, list(group))

    # Metrics generation    
    makeMetrics = input("Do you want to generate the metrics for the selected models? Y/n: ")
    if makeMetrics.lower() == 'y':
        await generateMetrics(env, nodes_to_process)
        #await generateMetrics(env, selected_nodes)
    # Metrics generation

    #print("Success! Make sure to run `dbt docs generate`.")
    run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)
    return 0

async def async_main(argv):
    await main(argv)

def run_async_main():
    if sys.platform == "win32" and sys.version_info >= (3, 8):        
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(async_main(sys.argv[1:]))

if __name__ == "__main__":
    run_async_main(sys.argv[1:])    
    run_async_main()