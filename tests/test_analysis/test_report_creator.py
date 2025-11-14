# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pandas as pd
import pytest
from matplotlib.figure import Figure

from ada_verona.analysis.report_creator import ReportCreator  # Replace with the actual module name


@pytest.fixture
def sample_df():
    # Sample data similar to what your class expects
    return pd.DataFrame({
        "epsilon_value": [0.1, 0.2, 0.15, 0.3, 0.05, 0.25],
        "network": ["net1", "net1", "net2", "net2", "net1", "net2"],
        "smallest_sat_value": [0.05, 0.1, 0.1, 0.2, 0.02, 0.15]
    })

def test_create_hist_figure(sample_df):
    rc = ReportCreator(sample_df)
    fig = rc.create_hist_figure()
    assert isinstance(fig, Figure)

def test_create_box_figure(sample_df):
    rc = ReportCreator(sample_df)
    fig = rc.create_box_figure()
    assert isinstance(fig, Figure)

def test_create_kde_figure(sample_df):
    rc = ReportCreator(sample_df)
    fig = rc.create_kde_figure()
    assert isinstance(fig, Figure)

def test_create_ecdf_figure(sample_df):
    rc = ReportCreator(sample_df)
    fig = rc.create_ecdf_figure()
    assert isinstance(fig, Figure)

def test_create_anneplot(sample_df):
    rc = ReportCreator(sample_df)
    ax = rc.create_anneplot()
    # The returned object should be a matplotlib Axes
    from matplotlib.axes import Axes
    assert isinstance(ax, Axes)