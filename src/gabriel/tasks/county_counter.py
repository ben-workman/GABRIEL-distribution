from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Sequence

import pandas as pd

from .elo import EloConfig, EloRater
from .regional import Regional, RegionalConfig
from ..utils import Teleprompter, create_county_choropleth


def _years_to_slices(years: Sequence[int | str]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for y in years:
        y = int(y)
        out.append((f"y{y}", f"{y:04d}-01-01", f"{y:04d}-12-31"))
    return out

def _time_msg(start: Optional[str], end: Optional[str]) -> str:
    if not start or not end:
        return ""
    y1, y2 = start[:4], end[:4]
    if y1 == y2 and start.endswith("01-01") and end.endswith("12-31"):
        return (
            f"ONLY use material from the year {y1}. "
            f"Ignore sources clearly outside {y1}. "
            "If evidence is sparse, say so explicitly."
        )
    return (
        f"ONLY use material created/published between {start} and {end} (inclusive). "
        "Ignore sources clearly outside this span. "
        "If evidence is sparse, state that explicitly."
    )
class RegionCounter:
    def __init__(
        self,
        df: pd.DataFrame,
        region_col: str,
        topics: List[str],
        *,
        years: Optional[Sequence[int | str]] = None,
        time_slices: Optional[List[Tuple[str, str, str]]] = None,
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
        elo_axis: str = "regions",
    ) -> None:
        self.df = df.copy()
        self.region_col = region_col
        self.geo_id_col = geo_id_col
        self.topics = topics
        if time_slices is None and years is not None:
            self.time_slices = _years_to_slices(years)
        else:
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
        self.elo_axis = elo_axis
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
        self,
        slice_label: Optional[str],
        start: Optional[str],
        end: Optional[str],
        reset_files: bool,
        run_elo: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        time_block = _time_msg(start, end)
        run_name = f"regional_{slice_label}" if slice_label else "regional_base"
        reg_cfg = RegionalConfig(
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=(
                f"{self.additional_instructions}\n{time_block}".strip()
                if time_block else self.additional_instructions
            ),
            additional_guidelines=self.additional_guidelines,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=False,
            save_dir=self.save_path,
            run_name=run_name,
        )
        regional = Regional(self.df, self.region_col, self.topics, reg_cfg)
        reports_df = await regional.run(reset_files=reset_files)
        if not run_elo:
            empty_results = reports_df[["region"]].copy().astype({"region": str})
            return empty_results, reports_df
        results = reports_df[["region"]].copy().astype({"region": str})
        for topic in self.topics:
            df_topic = reports_df[["region", topic]].rename(
                columns={"region": "identifier", topic: "text"}
            )
            attrs = list(self.elo_attributes.keys()) if self.elo_attributes else [topic]
            elo_instr_full = (
                f"{self.elo_instructions}\n{time_block}".strip()
                if time_block else self.elo_instructions
            )
            cfg = EloConfig(
                attributes=attrs,
                n_rounds=self.n_elo_rounds,
                n_parallels=self.n_parallels,
                model=self.model_elo,
                save_dir=self.save_path,
                run_name=f"elo_{topic}_{slice_label or 'base'}",
                use_dummy=self.use_dummy,
                instructions=elo_instr_full,
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
                        left_on="region",
                        right_on="identifier",
                        how="left",
                    )
                    out_col = f"{col}__{slice_label}" if slice_label else col
                    results[out_col] = results[tmp]
                    results = results.drop(columns=["identifier", tmp])
            else:
                col = score_cols[0]
                tmp = "__tmp_topic"
                results = results.merge(
                    elo_df[["identifier", col]].rename(columns={col: tmp}),
                    left_on="region",
                    right_on="identifier",
                    how="left",
                )
                out_col = f"{topic}__{slice_label}" if slice_label else topic
                results[out_col] = results[tmp]
                results = results.drop(columns=["identifier", tmp])
        return results, reports_df

    async def _elo_across_slices(
        self,
        final_df: pd.DataFrame,
        reports_df: pd.DataFrame,
        reset_files: bool,
    ) -> pd.DataFrame:
        from ..utils.teleprompter import Teleprompter
        import gabriel, inspect
        from pathlib import Path
        tele = Teleprompter(Path(inspect.getfile(gabriel)).parent / "prompts")
        melted = (
            reports_df
            .melt(id_vars=["region", "time_slice"], var_name="topic", value_name="text")
            .dropna(subset=["text"])
        )
        if "time_slice" not in melted.columns:
            return final_df
        out = final_df.copy()
        for topic in self.topics:
            df_topic = (
                melted[melted["topic"] == topic]
                .assign(identifier=lambda d: d["region"].astype(str) + "|" + d["time_slice"].astype(str))
                [["identifier", "text"]]
                .drop_duplicates("identifier")
            )
            attrs = list(self.elo_attributes.keys()) if self.elo_attributes else [topic]
            cfg = EloConfig(
                attributes=attrs,
                n_rounds=self.n_elo_rounds,
                n_parallels=self.n_parallels,
                model=self.model_elo,
                save_dir=self.save_path,
                run_name=f"elo_across_slices_{topic}",
                use_dummy=self.use_dummy,
                instructions=self.elo_instructions,
                additional_guidelines=self.elo_guidelines,
                print_example_prompt=False,
                timeout=self.elo_timeout,
            )
            elo = EloRater(tele, cfg)
            elo_df = await elo.run(df_topic, text_col="text", id_col="identifier", reset_files=reset_files)
            elo_df[["region", "time_slice"]] = elo_df["identifier"].str.split("|", n=1, expand=True)
            score_cols = [c for c in elo_df.columns if c not in ("identifier", "text", "region", "time_slice")]
            for sc in score_cols:
                pivoted = elo_df.pivot(index="region", columns="time_slice", values=sc).reset_index()
                ren = {c: f"{sc}__{topic}__{c}" for c in pivoted.columns if c != "region"}
                pivoted = pivoted.rename(columns=ren)
                out = out.merge(pivoted, left_on="region", right_on="region", how="left")
        return out

    async def run(self, *, reset_files: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.time_slices:
            all_results = []
            all_reports = []
            run_elo_inside = (self.elo_axis == "regions")
            for label, start, end in self.time_slices:
                res, rep = await self._run_one_slice(label, start, end, reset_files, run_elo_inside)
                all_results.append(res)
                rep = rep.copy()
                rep["time_slice"] = label
                all_reports.append(rep)
            final = all_results[0]
            for other in all_results[1:]:
                final = final.merge(other, on="region", how="outer")
            reports_df = pd.concat(all_reports, ignore_index=True)
            if self.elo_axis == "slices":
                final = await self._elo_across_slices(final, reports_df, reset_files)
            elif self.elo_axis == "both":
                melted = reports_df.melt(
                    id_vars=["region", "time_slice"],
                    var_name="topic",
                    value_name="text"
                ).dropna(subset=["text"])
                melted = melted.assign(
                    identifier=melted["region"] + "|" + melted["time_slice"]
                )
                df_items = melted[["identifier", "text"]].drop_duplicates("identifier")
                attrs = list(self.elo_attributes.keys()) if self.elo_attributes else self.topics
                cfg = EloConfig(
                    attributes=attrs,
                    n_rounds=self.n_elo_rounds,
                    n_parallels=self.n_parallels,
                    model=self.model_elo,
                    save_dir=self.save_path,
                    run_name="elo_joint_region_time",
                    use_dummy=self.use_dummy,
                    instructions=self.elo_instructions,
                    additional_guidelines=self.elo_guidelines,
                    timeout=self.elo_timeout,
                    print_example_prompt=False
                )
                rater = EloRater(self.tele, cfg)
                elo_df = await rater.run(
                    df_items,
                    text_col="text",
                    id_col="identifier",
                    reset_files=reset_files
                )
                final = elo_df.copy()
        else:
            res, rep = await self._run_one_slice(None, None, None, reset_files, self.elo_axis == "regions")
            final, reports_df = res, rep
            if self.elo_axis == "slices":
                final = await self._elo_across_slices(final, reports_df, reset_files)
        if self.geo_id_col and self.geo_id_col in self.df.columns and self.z_score_choropleth:
            merged = self.df[[self.region_col, self.geo_id_col]].drop_duplicates()
            merged[self.geo_id_col] = merged[self.geo_id_col].astype(str)
            final = final.merge(merged, left_on="region", right_on=self.region_col)
            value_cols = list(self.elo_attributes.keys()) if self.elo_attributes else self.topics
            for base in value_cols:
                cols = [c for c in final.columns if c.startswith(base)]
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
