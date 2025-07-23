from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple 

import pandas as pd

from .elo import EloConfig, EloRater
from .regional import Regional, RegionalConfig
from ..utils import Teleprompter, create_county_choropleth
class CountyCounter:
    """Run regional analysis on counties and rate them via Elo."""

    def __init__(
        self,
        df: pd.DataFrame,
        county_col: str,
        topics: List[str],
        *,
        fips_col: Optional[str] = None,
        save_dir: str = os.path.expanduser("~/Documents/runs"),
        run_name: Optional[str] = None,
        model_regional: str = "o4-mini",
        model_elo: str = "o4-mini",
        reasoning_effort: str = "medium",
        search_context_size: str = "medium",
        n_parallels: int = 400,
        n_elo_rounds: int = 15,
        elo_timeout: float = 60.0,
        use_dummy: bool = False,
        additional_instructions: str = "",
        elo_instructions: str = "",
        additional_guidelines: str = "",
        elo_guidelines: str = "",
        z_score_choropleth: bool = True,
        elo_attributes: Dict[str, str] | None = None,
    ) -> None:
        self.df = df.copy()
        self.county_col = county_col
        self.fips_col = fips_col
        self.topics = topics
        self.model_regional = model_regional
        self.model_elo = model_elo
        self.n_parallels = n_parallels
        self.n_elo_rounds = n_elo_rounds
        self.use_dummy = use_dummy
        self.additional_instructions = additional_instructions
        self.elo_instructions = elo_instructions
        self.additional_guidelines = additional_guidelines
        self.elo_guidelines = elo_guidelines
        self.reasoning_effort = reasoning_effort
        self.search_context_size = search_context_size
        self.z_score_choropleth = z_score_choropleth
        self.elo_attributes = elo_attributes
        self.elo_timeout = elo_timeout

        run_name = run_name or f"county_counter_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_path = os.path.join(save_dir, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        reg_cfg = RegionalConfig(
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=self.additional_instructions,
            additional_guidelines=self.additional_guidelines,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=True,
            save_dir=self.save_path,
            run_name="regional",
        )
        self.regional = Regional(self.df, self.county_col, self.topics, reg_cfg)
        self.tele = Teleprompter()

    async def run(self, *, reset_files: bool = False) -> pd.DataFrame:
        reports_df = await self.regional.run(reset_files=reset_files)
        results = reports_df[["region"]].copy()
        results["region"] = results["region"].astype(str)

        for topic in self.topics:
            df_topic = reports_df[["region", topic]].rename(
                columns={"region": "identifier", topic: "text"}
            )

            attributes = list(self.elo_attributes.keys()) if self.elo_attributes else [topic]

            cfg = EloConfig(
                attributes=attributes,
                n_rounds=self.n_elo_rounds,
                n_parallels=self.n_parallels,
                model=self.model_elo,
                save_dir=self.save_path,
                run_name=f"elo_{topic}",
                use_dummy=self.use_dummy,
                instructions=self.elo_instructions,
                additional_guidelines=self.elo_guidelines,
                print_example_prompt=False,
                timeout=self.elo_timeout,
            )

            rater = EloRater(self.tele, cfg)
            elo_df = await rater.run(
                df_topic, text_col="text", id_col="identifier", reset_files=reset_files
            )
            elo_df["identifier"] = elo_df["identifier"].astype(str)

            if self.elo_attributes:
                for attr in [k for k in elo_df.columns if k not in ("identifier", "text")]:
                    temp_col = f"_elo_temp_{attr}"
                    results = results.merge(
                        elo_df[["identifier", attr]].rename(columns={attr: temp_col}),
                        left_on="region",
                        right_on="identifier",
                        how="left",
                    )
                    results[attr] = results[temp_col]
                    results = results.drop(columns=["identifier", temp_col])
            else:
                temp_col = f"_elo_temp_{topic}"
                results = results.merge(
                    elo_df[["identifier", topic]].rename(columns={topic: temp_col}),
                    left_on="region",
                    right_on="identifier",
                    how="left",
                )
                results[topic] = results[temp_col]
                results = results.drop(columns=["identifier", temp_col])

        if self.fips_col and self.fips_col in self.df.columns:
            merged = self.df[[self.county_col, self.fips_col]].drop_duplicates()
            merged[self.fips_col] = merged[self.fips_col].astype(str).str.zfill(5)
            results = results.merge(
                merged, left_on="region", right_on=self.county_col
            )
            if self.elo_attributes:
                for attr in self.elo_attributes.keys():
                    map_path = os.path.join(self.save_path, f"map_{attr}.html")
                    create_county_choropleth(
                        results,
                        fips_col=self.fips_col,
                        value_col=attr,
                        title=f"ELO Rating for {attr}",
                        save_path=map_path,
                        z_score=self.z_score_choropleth,
                    )
            else:
                for topic in self.topics:
                    map_path = os.path.join(self.save_path, f"map_{topic}.html")
                    create_county_choropleth(
                        results,
                        fips_col=self.fips_col,
                        value_col=topic,
                        title=f"ELO Rating for {topic}",
                        save_path=map_path,
                        z_score=self.z_score_choropleth,
                    )

        results.to_csv(os.path.join(self.save_path, "county_elo.csv"), index=False)
        return results
class RegionCounter:
    """
    Generalized CountyCounter.
    Optionally segment by time periods if `time_slices` is provided.

    Parameters
    ----------
    df : DataFrame with your regions
    region_col : str, column name for region label
    topics : list of topics
    time_slices : optional list of tuples (label, start_str, end_str). If present,
                  separate Regional/Elo runs are done per slice and columns are suffixed "__<label>"
    geo_id_col : optional column for mapping (FIPS etc.)
    ...
    Returns
    -------
    results_df, reports_df  (both DataFrames)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        region_col: str,
        topics: List[str],
        *,
        time_slices: Optional[List[Tuple[str, str, str]]] = None,  # (label, start, end)
        geo_id_col: Optional[str] = None,
        save_dir: str = os.path.expanduser("~/Documents/runs"),
        run_name: Optional[str] = None,
        model_regional: str = "o4-mini",
        model_elo: str = "o4-mini",
        reasoning_effort: str = "medium",
        search_context_size: str = "medium",
        n_parallels: int = 400,
        n_elo_rounds: int = 15,
        elo_timeout: float = 60.0,
        use_dummy: bool = False,
        additional_instructions: str = "",
        elo_instructions: str = "",
        additional_guidelines: str = "",
        elo_guidelines: str = "",
        z_score_choropleth: bool = False,
        elo_attributes: Dict[str, str] | None = None,
        print_example_prompt: bool = False,
    ) -> None:
        self.df = df.copy()
        self.region_col = region_col
        self.geo_id_col = geo_id_col
        self.topics = topics
        self.time_slices = time_slices

        self.model_regional = model_regional
        self.model_elo = model_elo
        self.n_parallels = n_parallels
        self.n_elo_rounds = n_elo_rounds
        self.elo_timeout = elo_timeout
        self.use_dummy = use_dummy

        self.additional_instructions = additional_instructions
        self.additional_guidelines = additional_guidelines
        self.elo_instructions = elo_instructions
        self.elo_guidelines = elo_guidelines

        self.reasoning_effort = reasoning_effort
        self.search_context_size = search_context_size
        self.z_score_choropleth = z_score_choropleth
        self.elo_attributes = elo_attributes

        run_name = run_name or f"region_counter_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_path = os.path.join(save_dir, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.tele = Teleprompter()

        self.reg_cfg_base = RegionalConfig(
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=self.additional_instructions,
            additional_guidelines=self.additional_guidelines,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=print_example_prompt,
            save_dir=self.save_path,
            run_name="regional",
        )

    async def _run_one_slice(
        self, slice_label: Optional[str], start: Optional[str], end: Optional[str], reset_files: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # adjust RegionalConfig
        if slice_label:
            run_name = f"regional_{slice_label}"
            add_instr = (
                f"{self.additional_instructions}\nTime window: {start} to {end}. "
                "Ignore sources clearly outside this range."
            ).strip()
        else:
            run_name = "regional_base"
            add_instr = self.additional_instructions

        reg_cfg = RegionalConfig(
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=add_instr,
            additional_guidelines=self.additional_guidelines,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=False,
            save_dir=self.save_path,
            run_name=run_name,
        )

        regional = Regional(self.df, self.region_col, self.topics, reg_cfg)
        reports_df = await regional.run(reset_files=reset_files)
        results = reports_df[["region"]].copy().astype({"region": str})

        # Elo
        for topic in self.topics:
            df_topic = reports_df[["region", topic]].rename(columns={"region": "identifier", topic: "text"})

            attrs = list(self.elo_attributes.keys()) if self.elo_attributes else [topic]

            cfg = EloConfig(
                attributes=attrs,
                n_rounds=self.n_elo_rounds,
                n_parallels=self.n_parallels,
                model=self.model_elo,
                save_dir=self.save_path,
                run_name=f"elo_{topic}_{slice_label or 'base'}",
                use_dummy=self.use_dummy,
                instructions=self.elo_instructions,
                additional_guidelines=self.elo_guidelines,
                print_example_prompt=False,
                timeout=self.elo_timeout,
            )

            rater = EloRater(self.tele, cfg)
            elo_df = await rater.run(df_topic, text_col="text", id_col="identifier", reset_files=reset_files)
            elo_df["identifier"] = elo_df["identifier"].astype(str)

            score_cols = [c for c in elo_df.columns if c not in ("identifier", "text")]
            if self.elo_attributes:
                attr_list = list(self.elo_attributes.keys())
                if score_cols != attr_list and len(score_cols) == len(attr_list):
                    elo_df = elo_df.rename(columns=dict(zip(score_cols, attr_list)))
                for col in [c for c in elo_df.columns if c not in ("identifier", "text")]:
                    tmp = f"__tmp_{col}"
                    results = results.merge(
                        elo_df[["identifier", col]].rename(columns={col: tmp}),
                        left_on="region", right_on="identifier", how="left"
                    )
                    out_col = f"{col}__{slice_label}" if slice_label else col
                    results[out_col] = results[tmp]
                    results = results.drop(columns=["identifier", tmp])
            else:
                col = score_cols[0]
                tmp = "__tmp_topic"
                results = results.merge(
                    elo_df[["identifier", col]].rename(columns={col: tmp}),
                    left_on="region", right_on="identifier", how="left"
                )
                out_col = f"{topic}__{slice_label}" if slice_label else topic
                results[out_col] = results[tmp]
                results = results.drop(columns=["identifier", tmp])

        return results, reports_df

    async def run(self, *, reset_files: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.time_slices:
            all_results = []
            all_reports = []
            for label, start, end in self.time_slices:
                res, rep = await self._run_one_slice(label, start, end, reset_files)
                all_results.append(res)
                rep = rep.copy()
                rep["time_slice"] = label
                all_reports.append(rep)
            # merge results on region
            final = all_results[0]
            for other in all_results[1:]:
                final = final.merge(other, on="region", how="outer")
            reports_df = pd.concat(all_reports, ignore_index=True)
        else:
            final, reports_df = await self._run_one_slice(None, None, None, reset_files)

        if self.geo_id_col and self.geo_id_col in self.df.columns and self.z_score_choropleth:
            merged = self.df[[self.region_col, self.geo_id_col]].drop_duplicates()
            merged[self.geo_id_col] = merged[self.geo_id_col].astype(str)
            final = final.merge(merged, left_on="region", right_on=self.region_col)

            value_cols = (
                list(self.elo_attributes.keys()) if self.elo_attributes else self.topics
            )
            for col in value_cols:
                cols = [c for c in final.columns if c.startswith(col)]
                for cc in cols:
                    map_path = os.path.join(self.save_path, f"map_{cc}.html")
                    create_county_choropleth(
                        final,
                        fips_col=self.geo_id_col,
                        value_col=cc,
                        title=f"Elo Rating for {cc}",
                        save_path=map_path,
                        z_score=True,
                    )

        final.to_csv(os.path.join(self.save_path, "region_elo.csv"), index=False)
        reports_df.to_csv(os.path.join(self.save_path, "regional_reports.csv"), index=False)
        return final, reports_df
