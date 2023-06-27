import streamlit as st
import apps.app_TB as app_TB
import apps.app_face as app_face
import apps.app_reid as app_reid
import apps.app_au as app_au

# define app pages
app_pages = {
    app_TB.app_dict['name']: app_TB.app_dict['pages'],
    app_face.app_dict['name']: app_face.app_dict['pages'],
    app_au.app_dict['name']: app_au.app_dict['pages'],
    app_reid.app_dict['name']: app_reid.app_dict['pages']
}

# generic inputs
st.sidebar.markdown('## Navigation')
app_selected = st.sidebar.selectbox('选择应用', list(app_pages.keys()))
app_page = st.sidebar.selectbox('配置应用', app_pages[app_selected])

# app specific inputs
st.sidebar.markdown('## Inputs')
if app_selected == app_TB.app_dict['name']:
    inputs = app_TB.create_inputs(app_page)
elif app_selected == app_face.app_dict['name']:
    inputs = app_face.create_inputs(app_page)
elif app_selected == app_au.app_dict['name']:
    inputs = app_au.create_inputs(app_page)
elif app_selected == app_reid.app_dict['name']:
    inputs = app_reid.create_inputs(app_page)
# button to run app
app_run = st.sidebar.button('Run')

# render results
if app_run or app_page:
    if app_selected == app_TB.app_dict['name']:
        app_TB.render_results(inputs,app_page)
    elif app_selected == app_face.app_dict['name']:
        app_face.render_results(inputs,app_page)
    elif app_selected == app_au.app_dict['name']:
        app_au.render_results(inputs,app_page)
    elif app_selected == app_reid.app_dict['name']:
        app_reid.render_results(inputs,app_page)
