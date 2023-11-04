### 模板

    --副本进入
    local function _onEnter(player, dg)
        LOGD("======== challengeDragon on_enter  ========")
        local nNowTime = edi.general:get_now_seconds()
        local nEndTime = mod.nEndTime or (nNowTime + 3000) --gm刷新lua用
        --打开活动hud 传入活动结束时间
        local tabParam =
        {
            nTimeStamp = tonumber(nEndTime) - nNowTime,
        }
        open_ssr_cmd(player, M.HUD_SSRID , tabParam)
    end
    GameEvent.add(EventCfg.on_enter_dg, _onEnter, M.NAME)