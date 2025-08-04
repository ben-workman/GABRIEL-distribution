from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Sequence, Union

import pandas as pd

from .elo import EloConfig, EloRater
from .regional import Regional, RegionalConfig
from ..utils import create_county_choropleth


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
        return f"ONLY use material from the year {y1}. Ignore sources clearly outside {y1}. If evidence is sparse, say so explicitly."
    return f"ONLY use material created/published between {start} and {end} (inclusive). Ignore sources clearly outside this span. If evidence is sparse, state that explicitly."


class CountyCounter:
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
        elo_attributes: Optional[Dict] = None,
        years: Optional[Sequence[int | str]] = None,
        time_slices: Optional[List[Tuple[str, str, str]]] = None,
        elo_axis: str = "regions",
        return_reports: bool = False,
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
        self.elo_axis = elo_axis
        self.return_reports = return_reports

        if time_slices is None:
            if years is not None:
                self.time_slices = _years_to_slices(years)
            else:
                self.time_slices = None
        else:
            processed: List[Tuple[str, str, str]] = []
            for idx, ts in enumerate(time_slices):
                if len(ts) == 3:
                    processed.append(ts)
                else:
                    start, end = ts
                    processed.append((f"slice_{idx+1}", start, end))
            self.time_slices = processed

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

    async def _run_one_slice(
        self,
        slice_label: Optional[str],
        start: Optional[str],
        end: Optional[str],
        reset_files: bool,
        run_elo: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        time_block = _time_msg(start, end)
        ai = (f"{self.additional_instructions}\n{time_block}").strip() if time_block else self.additional_instructions

        reg_cfg = RegionalConfig(
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=ai,
            additional_guidelines=self.additional_guidelines,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=False,
            save_dir=self.save_path,
            run_name=f"regional_{slice_label}" if slice_label else "regional",
        )
        regional = Regional(self.df, self.county_col, self.topics, reg_cfg)
        reports_df = await regional.run(reset_files=reset_files)

        results = reports_df[["region"]].copy().astype({"region": str})
        if not run_elo:
            return results, reports_df

        for topic in self.topics:
            df_topic = reports_df[["region", topic]].rename(columns={"region": "identifier", topic: "text"})
            attributes = self.elo_attributes if self.elo_attributes else [topic]
            cfg = EloConfig(
                attributes=attributes,
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
            rater = EloRater(cfg)
            elo_df = await rater.run(df_topic, text_col="text", id_col="identifier", reset_files=reset_files)
            elo_df["identifier"] = elo_df["identifier"].astype(str)
            results["region"] = results["region"].astype(str)

            if self.elo_attributes:
                for attr in [k for k in elo_df.columns if k not in ("identifier", "text")]:
                    tmp = f"_elo_tmp_{attr}"
                    results = results.merge(
                        elo_df[["identifier", attr]].rename(columns={attr: tmp}),
                        left_on="region",
                        right_on="identifier",
                        how="left",
                    )
                    out_col = attr if not slice_label else f"{attr}__{slice_label}"
                    results[out_col] = results[tmp]
                    results = results.drop(columns=["identifier", tmp])
            else:
                tmp = f"_elo_tmp_{topic}"
                results = results.merge(
                    elo_df[["identifier", topic]].rename(columns={topic: tmp}),
                    left_on="region",
                    right_on="identifier",
                    how="left",
                )
                out_col = topic if not slice_label else f"{topic}__{slice_label}"
                results[out_col] = results[tmp]
                results = results.drop(columns=["identifier", tmp])

        return results, reports_df

    async def _elo_across_slices(
        self,
        reports_df: pd.DataFrame,
        reset_files: bool,
    ) -> pd.DataFrame:
        melted = (
            reports_df
            .melt(id_vars=["region", "time_slice"], var_name="topic", value_name="text")
            .dropna(subset=["text"])
        )
        out = melted[["region", "time_slice"]].drop_duplicates().astype({"region": str, "time_slice": str})
        for topic in self.topics:
            df_topic = (
                melted[melted["topic"] == topic]
                .assign(identifier=lambda d: d["region"].astype(str) + "|" + d["time_slice"].astype(str))
                [["identifier", "text"]]
                .drop_duplicates("identifier")
            )
            attributes = self.elo_attributes if self.elo_attributes else [topic]
            cfg = EloConfig(
                attributes=attributes,
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
            elo = EloRater(cfg)
            elo_df = await elo.run(df_topic, text_col="text", id_col="identifier", reset_files=reset_files)
            elo_df[["region", "time_slice"]] = elo_df["identifier"].str.split("|", n=1, expand=True)
            score_cols = [c for c in elo_df.columns if c not in ("identifier", "text", "region", "time_slice")]
            topic_df = elo_df[["region", "time_slice"] + score_cols].rename(columns={c: f"{c}__{topic}" for c in score_cols})
            out = out.merge(topic_df, on=["region", "time_slice"], how="left")
        return out

    async def run(self, *, reset_files: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        if self.time_slices:
            runs: List[pd.DataFrame] = []
            reports_all: List[pd.DataFrame] = []
            run_elo_inside = (self.elo_axis == "regions")
            for label, start, end in self.time_slices:
                res, rep = await self._run_one_slice(label, start, end, reset_files, run_elo_inside)
                runs.append(res)
                rep = rep.copy()
                rep["time_slice"] = label
                reports_all.append(rep)

            final = runs[0]
            for other in runs[1:]:
                final = final.merge(other, on="region", how="outer")
            reports_df = pd.concat(reports_all, ignore_index=True)

            if self.elo_axis in ("slices", "both"):
                across = await self._elo_across_slices(reports_df, reset_files)
                final = final.merge(across, on="region", how="left")
        else:
            reports_df = await self.regional.run(reset_files=reset_files)
            final = reports_df[["region"]].copy()
            for topic in self.topics:
                df_topic = reports_df[["region", topic]].rename(columns={"region": "identifier", topic: "text"})
                attributes = self.elo_attributes if self.elo_attributes else [topic]
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
                rater = EloRater(cfg)
                elo_df = await rater.run(df_topic, text_col="text", id_col="identifier", reset_files=reset_files)
                elo_df["identifier"] = elo_df["identifier"].astype(str)
                final["region"] = final["region"].astype(str)

                if self.elo_attributes:
                    for attr in [k for k in elo_df.columns if k not in ("identifier", "text")]:
                        tmp = f"_elo_tmp_{attr}"
                        final = final.merge(
                            elo_df[["identifier", attr]].rename(columns={attr: tmp}),
                            left_on="region",
                            right_on="identifier",
                            how="left",
                        )
                        final[attr] = final[tmp]
                        final = final.drop(columns=["identifier", tmp])
                else:
                    tmp = f"_elo_tmp_{topic}"
                    final = final.merge(
                        elo_df[["identifier", topic]].rename(columns={topic: tmp}),
                        left_on="region",
                        right_on="identifier",
                        how="left",
                    )
                    final[topic] = final[tmp]
                    final = final.drop(columns=["identifier", tmp])

        if self.fips_col and self.fips_col in self.df.columns:
            merged = self.df[[self.county_col, self.fips_col]].drop_duplicates()
            merged[self.fips_col] = merged[self.fips_col].astype(str).str.zfill(5)
            final = final.merge(merged, left_on="region", right_on=self.county_col, how="left")
            if self.elo_attributes:
                for attr in self.elo_attributes.keys():
                    path = os.path.join(self.save_path, f"map_{attr}.html")
                    create_county_choropleth(final, fips_col=self.fips_col, value_col=attr, title=f"ELO Rating for {attr}", save_path=path, z_score=self.z_score_choropleth)
            else:
                for topic in self.topics:
                    path = os.path.join(self.save_path, f"map_{topic}.html")
                    create_county_choropleth(final, fips_col=self.fips_col, value_col=topic, title=f"ELO Rating for {topic}", save_path=path, z_score=self.z_score_choropleth)

        final.to_csv(os.path.join(self.save_path, "region_elo.csv"), index=False)
        reports_df.to_csv(os.path.join(self.save_path, "regional_reports.csv"), index=False)
        return (final, reports_df) if self.return_reports else final


class RegionCounter(CountyCounter):
    def __init__(self, df: pd.DataFrame, region_col: str, topics: List[str], *, geo_id_col: Optional[str] = None, **kwargs) -> None:
        super().__init__(df=df, county_col=region_col, topics=topics, fips_col=geo_id_col, **kwargs)
