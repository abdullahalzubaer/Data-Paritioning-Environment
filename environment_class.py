"""
Formulating the problem of data partitioning (for TPC-H dataset) in reinforcement learning framework.
This is an working environment suitable for applyingh DRL agents with options tailored towards
improving its efficiency such as caching (file and runtime caching)
----
Note: This is part of my Master's Thesis reinforcement learning's environment implementation.

This library is needed to install
!pip install pyspark

Tested on Google Colaboratory
"""

import copy
import os
import pickle
import signal
import time

import gym
import numpy as np
from pyspark.sql import SparkSession

import queries

spark = (
    SparkSession.builder.master("local[*]")
    .appName("TPCH")
    .config("spark.driver.memory", "60g")
    .config(
        "spark.sql.hive.filesourcePartitionFileCacheSize", 256 * 1024 * 1024 * 10000
    )
    .getOrCreate()
)


class Environment(gym.Env):

    def __init__(self):
        self.files_path = "/content/drive/MyDrive/EnvToyThesis/toypartitioningenv/"
        self.dict_path = "/content/"
        if not os.path.exists(
            self.dict_path + "non_partitioned_data_query_executiion_runtime.pickle"
        ):
            self.dictionary_filelocation_track = dict()
        else:
            with open(
                self.dict_path + "dictionary_filelocation_track.pickle", "rb"
            ) as f:
                self.dictionary_filelocation_track = pickle.load(f)

        if not os.path.exists(self.dict_path + "invalid_actions.pickle"):
            self.invalid_actions = set()
        else:
            with open(self.dict_path + "invalid_actions.pickle", "rb") as f:
                self.invalid_actions = pickle.load(f)

        if not os.path.exists(self.dict_path + "dictionary_runtime_track.pickle"):
            self.dictionary_runtime_track = dict()
        else:
            with open(self.dict_path + "dictionary_runtime_track.pickle", "rb") as f:
                self.dictionary_runtime_track = pickle.load(f)

        schema = {
            "nation": sorted(
                set(["n_nationkey", "n_name", "n_regionkey", "n_comment"])
            ),
            "region": sorted(set(["r_regionkey", "r_name", "r_comment"])),
            "part": sorted(
                set(
                    [
                        "p_partkey",
                        "p_name",
                        "p_mfgr",
                        "p_brand",
                        "p_type",
                        "p_size",
                        "p_container",
                        "p_retailprice",
                        "p_comment",
                    ]
                )
            ),
            "supplier": sorted(
                set(
                    [
                        "s_suppkey",
                        "s_name",
                        "s_address",
                        "s_nationkey",
                        "s_phone",
                        "s_acctbal",
                        "s_comment",
                    ]
                )
            ),
            "partsupp": sorted(
                set(
                    [
                        "ps_partkey",
                        "ps_suppkey",
                        "ps_availqty",
                        "ps_supplycost",
                        "ps_comment",
                    ]
                )
            ),
            "customer": sorted(
                set(
                    [
                        "c_custkey",
                        "c_name",
                        "c_address",
                        "c_nationkey",
                        "c_phone",
                        "c_acctbal",
                        "c_mktsegment",
                        "c_comment",
                    ]
                )
            ),
            "orders": sorted(
                set(
                    [
                        "o_orderkey",
                        "o_custkey",
                        "o_orderstatus",
                        "o_totalprice",
                        "o_orderdate",
                        "o_orderpriority",
                        "o_clerk",
                        "o_shippriority",
                        "o_comment",
                    ]
                )
            ),
            "lineitem": sorted(
                set(
                    [
                        "l_orderkey",
                        "l_partkey",
                        "l_suppkey",
                        "l_linenumber",
                        "l_quantity",
                        "l_extendedprice",
                        "l_discount",
                        "l_tax",
                        "l_returnflag",
                        "l_linestatus",
                        "l_shipdate",
                        "l_commitdate",
                        "l_receiptdate",
                        "l_shipinstruct",
                        "l_shipmode",
                        "l_comment",
                    ]
                )
            ),
        }

        total_col = 0
        for i in schema.keys():
            for k in schema[i]:
                total_col = total_col + 1

        counter = 0
        col_to_pos = dict()

        for i_keys in sorted(schema.keys()):
            for i in schema[i_keys]:
                col_to_pos[i] = counter
                counter += 1

        action_list = list()
        for i_keys in sorted(schema.keys()):
            for i in schema[i_keys]:
                action_list.append(i)

        self.action_map_dictionary = dict(
            [(y, x) for x, y in enumerate(sorted(set(action_list)), start=0)]
        )

        state = np.zeros([total_col, total_col])

        table_find_to_partition = {
            "n": "nation",
            "r": "region",
            "p": "part",
            "s": "supplier",
            "ps": "partsupp",
            "c": "customer",
            "o": "orders",
            "l": "lineitem",
        }

        self.dictionary_runtime_track = self.dictionary_runtime_track
        self.action_track = list()

        self.dictionary_filelocation_track = self.dictionary_filelocation_track
        self.schema = schema
        self.col_to_pos = col_to_pos
        self.total_col = total_col
        self.state = state
        self.table_find_to_partition = table_find_to_partition
        self.action_space = gym.spaces.Discrete(self.total_col)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.total_col, self.total_col),
            dtype=np.float32,
        )

        # Creating the initial view of the all the tables when we initiate the environment class

        self.region_parquet_initial = self.files_path + "region_initial.parquet0"
        self.df_spark_region_parquet = spark.read.parquet(self.region_parquet_initial)
        self.df_spark_region_parquet.createOrReplaceTempView("region_parquet")

        self.nation_parquet_initial = self.files_path + "nation_initial.parquet0"
        self.df_spark_nation_parquet = spark.read.parquet(self.nation_parquet_initial)
        self.df_spark_nation_parquet.createOrReplaceTempView("nation_parquet")

        self.supplier_parquet_initial = self.files_path + "supplier_initial.parquet0"
        self.df_spark_supplier_parquet = spark.read.parquet(
            self.supplier_parquet_initial
        )
        self.df_spark_supplier_parquet.createOrReplaceTempView("supplier_parquet")

        self.partsupp_parquet_initial = self.files_path + "partsupp_initial.parquet0"
        self.df_spark_partsupp_parquet = spark.read.parquet(
            self.partsupp_parquet_initial
        )
        self.df_spark_partsupp_parquet.createOrReplaceTempView("partsupp_parquet")

        self.part_parquet_initial = self.files_path + "part_initial.parquet0"
        self.df_spark_part_parquet = spark.read.parquet(self.part_parquet_initial)
        self.df_spark_part_parquet.createOrReplaceTempView("part_parquet")

        self.customer_parquet_initial = self.files_path + "customer_initial.parquet0"
        self.df_spark_customer_parquet = spark.read.parquet(
            self.customer_parquet_initial
        )
        self.df_spark_customer_parquet.createOrReplaceTempView("customer_parquet")

        self.lineitem_parquet_initial = self.files_path + "lineitem_initial.parquet0"
        self.df_spark_lineitem_parquet = spark.read.parquet(
            self.lineitem_parquet_initial
        )
        self.df_spark_lineitem_parquet.createOrReplaceTempView("lineitem_parquet")

        self.orders_parquet_initial = self.files_path + "orders_initial.parquet0"
        self.df_spark_orders_parquet = spark.read.parquet(self.orders_parquet_initial)
        self.df_spark_orders_parquet.createOrReplaceTempView("orders_parquet")

        ##### Getting the runtime on non-partitioned data########

        # All the queries of TPCH, except query number 15

        self.q1 = queries.q1
        self.q2 = queries.q2
        self.q3 = queries.q3
        self.q4 = queries.q4
        self.q5 = queries.q5
        self.q6 = queries.q6
        self.q7 = queries.q7
        self.q8 = queries.q8
        self.q9 = queries.q9
        self.q10 = queries.q10
        self.q11 = queries.q11
        self.q12 = queries.q12
        self.q13 = queries.q13
        self.q14 = queries.q14
        self.q16 = queries.q16
        self.q17 = queries.q17
        self.q18 = queries.q18
        self.q19 = queries.q19
        self.q20 = queries.q20
        self.q21 = queries.q21
        self.q22 = queries.q22

        isExist = os.path.exists(
            self.dict_path + "non_partitioned_data_query_executiion_runtime.pickle"
        )

        if isExist == False:
            startTimeQuery = time.time()

            query_1 = spark.sql(self.q1).count()
            query_2 = spark.sql(self.q2).count()
            query_3 = spark.sql(self.q3).count()
            query_4 = spark.sql(self.q4).count()
            query_5 = spark.sql(self.q5).count()
            query_6 = spark.sql(self.q6).count()
            query_7 = spark.sql(self.q7).count()
            query_8 = spark.sql(self.q8).count()
            query_9 = spark.sql(self.q9).count()
            query_10 = spark.sql(self.q10).count()
            query_11 = spark.sql(self.q11).count()
            query_12 = spark.sql(self.q12).count()
            query_13 = spark.sql(self.q13).count()
            query_14 = spark.sql(self.q14).count()
            query_16 = spark.sql(self.q16).count()
            query_17 = spark.sql(self.q17).count()
            query_18 = spark.sql(self.q18).count()
            query_19 = spark.sql(self.q19).count()
            query_20 = spark.sql(self.q20).count()
            query_21 = spark.sql(self.q21).count()
            query_22 = spark.sql(self.q22).count()

            endTimeQuery = time.time()
            runtime_nonpartioned_data = endTimeQuery - startTimeQuery
            self.runtime_nonpartioned_data = runtime_nonpartioned_data
            with open(
                self.dict_path + "non_partitioned_data_query_executiion_runtime.pickle",
                "wb",
            ) as f:
                pickle.dump(self.runtime_nonpartioned_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            self.runtime_nonpartioned_data = pickle.load(
                open(
                    self.dict_path
                    + "non_partitioned_data_query_executiion_runtime.pickle",
                    "rb",
                )
            )

        self.reset()

    @property
    def _n_actions(self):

        return int(self.total_col)

    def _get_obs(self):

        return copy.deepcopy(np.expand_dims(self.state, axis=0))

    def _calculate_reward(self):

        rt = self.__getRuntime(self.action)
        self.stats = {"runtime": copy.deepcopy(rt)}
        self.reward = max(float(self.runtime_nonpartioned_data / rt - 1), 0)

        return self.reward

    def __getRuntime(self, action):

        self.action = action
        self.action_track.append(self.action)

        if next(
            (
                value
                for key, value in self.dictionary_runtime_track.items()
                if frozenset(self.action_track) == key
            ),
            None,
        ):
            self.runtime_from_dictionary = next(
                value
                for key, value in self.dictionary_runtime_track.items()
                if frozenset(self.action_track) == key
            )
            self.runtime_stats_dictionary["runtime"].append(
                self.runtime_from_dictionary
            )

            return self.runtime_from_dictionary

        else:

            self.first_letter_of_table = (self.action).split("_")[0]
            self.table_to_partition = copy.deepcopy(
                self.table_find_to_partition[self.first_letter_of_table]
            )

            ##########################################################-----tables-----############################################################################
            if next(
                (
                    value
                    for key, value in self.dictionary_filelocation_track.items()
                    if frozenset([self.action]) == key
                ),
                None,
            ):

                self.file_location = next(
                    value
                    for key, value in self.dictionary_filelocation_track.items()
                    if frozenset([self.action]) == key
                )
                self.df_spark_table_partitioned = spark.read.parquet(
                    " ".join(self.file_location)
                )
                self.df_spark_table_partitioned.createOrReplaceTempView(
                    self.table_to_partition + "_parquet"
                )

            else:
                if self.table_to_partition == "region":
                    self.df_spark_table = spark.read.parquet(
                        self.region_parquet_initial
                    )
                elif self.table_to_partition == "nation":
                    self.df_spark_table = spark.read.parquet(
                        self.nation_parquet_initial
                    )
                elif self.table_to_partition == "supplier":
                    self.df_spark_table = spark.read.parquet(
                        self.supplier_parquet_initial
                    )
                elif self.table_to_partition == "partsupp":
                    self.df_spark_table = spark.read.parquet(
                        self.partsupp_parquet_initial
                    )
                elif self.table_to_partition == "part":
                    self.df_spark_table = spark.read.parquet(self.part_parquet_initial)
                elif self.table_to_partition == "customer":
                    self.df_spark_table = spark.read.parquet(
                        self.customer_parquet_initial
                    )
                elif self.table_to_partition == "lineitem":
                    self.df_spark_table = spark.read.parquet(
                        self.lineitem_parquet_initial
                    )
                elif self.table_to_partition == "orders":
                    self.df_spark_table = spark.read.parquet(
                        self.orders_parquet_initial
                    )

                self.writing_partitioned_file = (
                    self.files_path
                    + self.table_to_partition
                    + "_initial.parquet {}".format(self.action)
                )
                if not os.path.exists(self.writing_partitioned_file):
                    self.df_spark_table.write.partitionBy(self.action).parquet(
                        self.writing_partitioned_file
                    )
                self.df_spark_table_partitioned = spark.read.parquet(
                    self.writing_partitioned_file
                )
                self.df_spark_table_partitioned.createOrReplaceTempView(
                    self.table_to_partition + "_parquet"
                )
                self.dictionary_filelocation_track[frozenset([self.action])] = [
                    self.writing_partitioned_file
                ]
                with open(
                    self.dict_path + "dictionary_filelocation_track.pickle", "wb"
                ) as f:
                    pickle.dump(
                        self.dictionary_filelocation_track, f, pickle.HIGHEST_PROTOCOL
                    )

            start_time_now = time.time()
            spark.sql(self.q1).count()
            spark.sql(self.q2).count()
            spark.sql(self.q3).count()
            spark.sql(self.q4).count()
            spark.sql(self.q5).count()
            spark.sql(self.q6).count()
            spark.sql(self.q7).count()
            spark.sql(self.q8).count()
            spark.sql(self.q9).count()
            spark.sql(self.q10).count()
            spark.sql(self.q11).count()
            spark.sql(self.q12).count()
            spark.sql(self.q13).count()
            spark.sql(self.q14).count()
            # spark.sql(self.q15).count()
            spark.sql(self.q16).count()
            spark.sql(self.q17).count()
            spark.sql(self.q18).count()
            spark.sql(self.q19).count()
            spark.sql(self.q20).count()
            spark.sql(self.q21).count()
            spark.sql(self.q22).count()

            self.total_runtime_partitioned_data = time.time() - start_time_now

            self.dictionary_runtime_track[
                frozenset(self.action_track)
            ] = self.total_runtime_partitioned_data

            with open(self.dict_path + "dictionary_runtime_track.pickle", "wb") as f:
                pickle.dump(self.dictionary_runtime_track, f, pickle.HIGHEST_PROTOCOL)

            """Runtime Stats"""
            self.runtime_stats_dictionary["runtime"].append(
                self.total_runtime_partitioned_data
            )

            return self.total_runtime_partitioned_data

    def reset(self):

        self.table_to_partition = None
        self.stats = {"runtime": copy.deepcopy(self.runtime_nonpartioned_data)}
        self.runtime_stats_dictionary = dict()
        self.runtime_stats_dictionary["runtime"] = list()
        self.number_of_steps_in_an_episode = 0
        self._init_action_mask()
        self.state = np.zeros([self.total_col, self.total_col])
        self.action_track = list()

        self.region_parquet_initial = self.files_path + "region_initial.parquet0"
        self.df_spark_region_parquet = spark.read.parquet(self.region_parquet_initial)
        self.df_spark_region_parquet.createOrReplaceTempView("region_parquet")

        self.nation_parquet_initial = self.files_path + "nation_initial.parquet0"
        self.df_spark_nation_parquet = spark.read.parquet(self.nation_parquet_initial)
        self.df_spark_nation_parquet.createOrReplaceTempView("nation_parquet")

        self.supplier_parquet_initial = self.files_path + "supplier_initial.parquet0"
        self.df_spark_supplier_parquet = spark.read.parquet(
            self.supplier_parquet_initial
        )
        self.df_spark_supplier_parquet.createOrReplaceTempView("supplier_parquet")

        self.partsupp_parquet_initial = self.files_path + "partsupp_initial.parquet0"
        self.df_spark_partsupp_parquet = spark.read.parquet(
            self.partsupp_parquet_initial
        )
        self.df_spark_partsupp_parquet.createOrReplaceTempView("partsupp_parquet")

        self.part_parquet_initial = self.files_path + "part_initial.parquet0"
        self.df_spark_part_parquet = spark.read.parquet(self.part_parquet_initial)
        self.df_spark_part_parquet.createOrReplaceTempView("part_parquet")

        self.lineitem_parquet_initial = self.files_path + "lineitem_initial.parquet0"
        self.df_spark_lineitem_parquet = spark.read.parquet(
            self.lineitem_parquet_initial
        )
        self.df_spark_lineitem_parquet.createOrReplaceTempView("lineitem_parquet")

        self.orders_parquet_initial = self.files_path + "orders_initial.parquet0"
        self.df_spark_orders_parquet = spark.read.parquet(self.orders_parquet_initial)
        self.df_spark_orders_parquet.createOrReplaceTempView("orders_parquet")

        self.customer_parquet_initial = self.files_path + "customer_initial.parquet0"
        self.df_spark_customer_parquet = spark.read.parquet(
            self.customer_parquet_initial
        )
        self.df_spark_customer_parquet.createOrReplaceTempView("customer_parquet")

        self._init_action_mask()
        self.number_of_steps_in_an_episode = 0

        self.table_to_partition = str()
        return copy.deepcopy(np.expand_dims(self.state, axis=0))

    def close(self,):
        return None

    def step(self, action):

        self.number_of_steps_in_an_episode += 1

        timeout_duration = 6000

        class TimeoutError(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutError()

        self.action = int(action)

        for key_action, val_action in (self.action_map_dictionary).items():
            if self.action == val_action:
                self.action = key_action
        user_chosen_action_index = self.col_to_pos[self.action]

        self.first_letter_of_table = (self.action).split("_")[0]
        self.table_to_partition = self.table_find_to_partition[
            self.first_letter_of_table
        ]

        if any(
            [
                self.state[self.col_to_pos[k]][self.col_to_pos[k]] == 1
                for k in self.schema[self.table_to_partition]
            ]
        ):

            self.state = self.state
            self.done = True
            self.reward = -1

            return (
                copy.deepcopy(np.expand_dims(self.state, axis=0)),
                float(self.reward),
                self.done,
                copy.deepcopy(self.stats),
            )

        else:

            self.done = False
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_duration)
            try:
                self.reward = self._calculate_reward()
                for k in self.schema[self.table_to_partition]:
                    self.state[self.col_to_pos[k]][self.col_to_pos[k]] = 1
            except TimeoutError as exc:
                print(exc)
                self.reward = -1
                self.done = True
                self.invalid_actions.add(int(action))
                with open(self.dict_path + "invalid_actions.pickle", "wb") as f:
                    pickle.dump(self.invalid_actions, f, pickle.HIGHEST_PROTOCOL)
            except Exception as exc:  # Maybe folder is empty from another timeout...
                print(exc)
                self.reward = -1
                self.done = True
                self.invalid_actions.add(int(action))
                with open(self.dict_path + "invalid_actions.pickle", "wb") as f:
                    pickle.dump(self.invalid_actions, f, pickle.HIGHEST_PROTOCOL)
            finally:
                signal.alarm(0)

            if self.number_of_steps_in_an_episode >= 1:  # Changed From 6 to 1
                self.done = True
            return (
                copy.deepcopy(np.expand_dims(self.state, axis=0)),
                float(self.reward),
                self.done,
                copy.deepcopy(self.stats),
            )

    def _init_action_mask(self):

        self.action_mask = np.ones(([self.total_col]), dtype=np.int32)
        for item in self.invalid_actions:
            self.action_mask[item] = 0
        return self.action_mask

    def _get_action_mask(self):
        if self.table_to_partition != None:
            try:
                for k in self.schema[self.table_to_partition]:
                    self.action_mask[self.action_map_dictionary[k]] = 0
            except:
                pass
        return copy.deepcopy(self.action_mask)

    def _is_done(self):
        if (
            self.number_of_steps_in_an_episode > 1
        ):  # Changed 5 TO 1, NOW WE HAVE 1 STEP IN ONE EPISODE
            return True
        else:
            return False
