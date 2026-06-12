
import os
import streamlit as st
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_common_sidebar():
    """Hàm này sẽ được gọi ở ĐẦU mỗi file trang để đồng bộ giao diện bên trái."""
    st.sidebar.markdown("---")  # Đường vạch ngăn cách với menu trang mặc định
    st.sidebar.image(
        "https://img.icons8.com/clouds/200/000000/weather.png", width=80
    )

    cfg = load_config()
    if cfg:
        st.sidebar.caption("📋 **Cấu hình Realtime:**")
        st.sidebar.caption(f"- Dự án: `{cfg['global']['project_name']}`")
        st.sidebar.caption(
            f"- Giai đoạn: `{cfg['data']['start_year']} - {cfg['data']['end_year']}`"
        )
        st.sidebar.caption(f"- Định dạng: `{cfg['data']['format'].upper()}`")
        st.sidebar.caption(f"- Thiết bị: `{cfg['global']['device'].upper()}`")
    else:
        st.sidebar.caption("⚠️ Chưa cấu hình hệ thống.")

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 AI Weather Forecasting")