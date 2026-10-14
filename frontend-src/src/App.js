import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import io from 'socket.io-client';
import axios from 'axios';
import './App.css';

const ACTIVE_TASK_KEY = 'motion_active_task_id';
const ACTIVE_FILE_KEY = 'motion_active_file_name';
const HISTORY_KEY = 'motion_history_records';

function resolveBackendBaseUrl() {
  const envUrl = process.env.REACT_APP_BACKEND_URL;
  if (envUrl) {
    return envUrl.replace(/\/$/, '');
  }

  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:5000';
  }

  const { protocol, hostname, port } = window.location;
  const devPorts = ['3000', '3001', '3002', '5173', '5174'];
  if (devPorts.includes(port) || ['localhost', '127.0.0.1', '0.0.0.0'].includes(hostname)) {
    return 'http://127.0.0.1:5000';
  }

  return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
}

const BACKEND_BASE_URL = resolveBackendBaseUrl();
axios.defaults.baseURL = BACKEND_BASE_URL;

function toAbsoluteUrl(url) {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${BACKEND_BASE_URL}${url}`;
}

function getFileKind(file) {
  if (!file) return '';
  return '视频';
}

function formatPercent(current, total) {
  if (!current || !total) return 0;
  return Math.min(100, Math.round((current / total) * 100));
}

function GlassVideoPlayer({ title, src, downloadName }) {
  const videoRef = useRef(null);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
    } else {
      v.pause();
    }
  };

  const rewind = () => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, v.currentTime - 10);
  };

  const capture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth || !v.videoHeight) return;
    const canvas = document.createElement('canvas');
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${downloadName || 'snapshot'}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  };

  const downloadVideo = async () => {
    if (!src) return;
    try {
      const response = await fetch(src);
      if (!response.ok) return;
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = `${downloadName || 'result'}.mp4`;
      a.click();
      URL.revokeObjectURL(objectUrl);
    } catch (e) {
      console.error(e);
    }
  };

  const fullScreen = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.requestFullscreen) {
      v.requestFullscreen();
    }
  };

  return (
    <section className="video-block">
      <div className="video-title">{title}</div>
      <video ref={videoRef} className="video-surface" src={src} controls />
      <div className="control-row">
        <button type="button" className="glass-btn" onClick={togglePlay}>播放/暂停</button>
        <button type="button" className="glass-btn" onClick={capture}>截图</button>
        <button type="button" className="glass-btn" onClick={rewind}>回看10秒</button>
        <button type="button" className="glass-btn" onClick={downloadVideo}>下载</button>
        <button type="button" className="glass-btn" onClick={fullScreen}>全屏</button>
      </div>
    </section>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [socketStatus, setSocketStatus] = useState('disconnected');
  const [socketError, setSocketError] = useState('');
  const [currentFrame, setCurrentFrame] = useState(null);
  const [currentStats, setCurrentStats] = useState(null);
  const [totalFrames, setTotalFrames] = useState(0);
  const [videoUrl, setVideoUrl] = useState('');
  const [logUrl, setLogUrl] = useState('');
  const [subVideosZipUrl, setSubVideosZipUrl] = useState('');
  const [summaryUrl, setSummaryUrl] = useState('');
  const [swingCsvUrl, setSwingCsvUrl] = useState('');
  const [swingEventsUrl, setSwingEventsUrl] = useState('');
  const [swingDiagUrl, setSwingDiagUrl] = useState('');
  const [diskTasks, setDiskTasks] = useState([]);
  const [diskTotalMb, setDiskTotalMb] = useState(0);
  const [diskUploadMb, setDiskUploadMb] = useState(0);
  const [subVideos, setSubVideos] = useState([]);
  const [selectedSubVideo, setSelectedSubVideo] = useState('');
  const [historyList, setHistoryList] = useState([]);
  const [idPoseSummary, setIdPoseSummary] = useState([]);
  const [error, setError] = useState('');
  const [processingHint, setProcessingHint] = useState('请上传视频后开始识别。');

  const socketRef = useRef(null);
  const taskRef = useRef(null);
  const selectedFileNameRef = useRef(selectedFileName);
  const selectedSubVideoRef = useRef(selectedSubVideo);

  useEffect(() => {
    taskRef.current = taskId;
  }, [taskId]);

  useEffect(() => {
    selectedFileNameRef.current = selectedFileName;
  }, [selectedFileName]);

  useEffect(() => {
    selectedSubVideoRef.current = selectedSubVideo;
  }, [selectedSubVideo]);

  useEffect(() => {
    const savedTaskId = sessionStorage.getItem(ACTIVE_TASK_KEY);
    const savedFileName = sessionStorage.getItem(ACTIVE_FILE_KEY);
    const savedHistory = localStorage.getItem(HISTORY_KEY);
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        if (Array.isArray(parsed)) {
          setHistoryList(parsed);
        }
      } catch (e) {
        console.error(e);
      }
    }
    if (savedFileName) {
      setSelectedFileName(savedFileName);
    }
    if (savedTaskId) {
      setTaskId(savedTaskId);
      setStatus('processing');
    }
  }, []);

  useEffect(() => {
    const socket = io(BACKEND_BASE_URL, { transports: ['websocket', 'polling'] });
    socketRef.current = socket;

    socket.on('connect', () => {
      setSocketStatus('connected');
      setSocketError('');
      if (taskRef.current) {
        socket.emit('join', { task_id: String(taskRef.current) });
      }
    });

    socket.on('disconnect', () => {
      setSocketStatus('disconnected');
    });

    socket.on('connect_error', (err) => {
      setSocketStatus('error');
      setSocketError(err?.message || '无法连接到后端 Socket 服务');
    });

    socket.on('frame', (data) => {
      if (String(data.task_id) !== String(taskRef.current)) return;
      setCurrentFrame(`data:image/jpeg;base64,${data.image}`);
      setCurrentStats(data.stats);
      setTotalFrames(data.total_frames || 0);
      setProcessingHint('正在生成实时处理结果...');
    });

    socket.on('complete', async (data) => {
      if (String(data.task_id) !== String(taskRef.current)) return;

      setVideoUrl(toAbsoluteUrl(data.video_url || ''));
      setLogUrl(toAbsoluteUrl(data.log_url || ''));
      setSubVideosZipUrl(toAbsoluteUrl(data.sub_videos_url || ''));
      setSummaryUrl(toAbsoluteUrl(data.summary_url || ''));
      setSwingCsvUrl(toAbsoluteUrl(data.swing_csv_url || ''));
      setSwingEventsUrl(toAbsoluteUrl(data.swing_events_url || ''));
      setSwingDiagUrl(toAbsoluteUrl(data.swing_diag_url || ''));
      setIdPoseSummary(data.id_pose_summary || []);
      setStatus('completed');
      loadDiskTasks();
      setProcessingHint('处理已完成，可查看最终结果。');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
      sessionStorage.removeItem(ACTIVE_FILE_KEY);

      try {
        const res = await axios.get(`/task/${data.task_id}/sub_videos`);
        const videos = (res.data.videos || []).map((item) => ({
          ...item,
          url: toAbsoluteUrl(item.url)
        }));
        setSubVideos(videos);
        setSelectedSubVideo(videos[0]?.url || '');
      } catch (e) {
        console.error(e);
      }

      setHistoryList((prev) => {
        const record = {
          id: String(data.task_id),
          name: selectedFileNameRef.current || `任务${data.task_id}`,
          time: new Date().toLocaleString(),
          status: 'completed',
          video: toAbsoluteUrl(data.video_url || ''),
          log: toAbsoluteUrl(data.log_url || ''),
          zip: toAbsoluteUrl(data.sub_videos_url || ''),
          summary: toAbsoluteUrl(data.summary_url || ''),
          swingCsv: toAbsoluteUrl(data.swing_csv_url || '')
        };
        const next = [record, ...prev.filter((item) => String(item.id) !== String(record.id))].slice(0, 50);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
        return next;
      });
    });

    socket.on('cancelled', (payload) => {
      if (String(payload.task_id) !== String(taskRef.current)) return;
      setStatus('cancelled');
      setProcessingHint('任务已取消。');
      setError(payload.message || '任务已取消');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
      sessionStorage.removeItem(ACTIVE_FILE_KEY);
    });

    socket.on('error', (payload) => {
      if (String(payload.task_id) !== String(taskRef.current)) return;
      setStatus('idle');
      setProcessingHint('处理失败，请检查上传文件和后端日志。');
      setError(payload.message || '处理失败');
      sessionStorage.removeItem(ACTIVE_TASK_KEY);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    if (taskId && socketRef.current?.connected) {
      socketRef.current.emit('join', { task_id: String(taskId) });
    }
  }, [taskId]);

  const upsertHistoryRecord = useCallback((record) => {
    setHistoryList((prev) => {
      const id = String(record.id);
      const existing = prev.find((item) => String(item.id) === id) || {};
      const payload = {
        id,
        name: record.name ?? existing.name ?? `任务${id}`,
        status: record.status ?? existing.status ?? 'processing',
        time: record.time ?? new Date().toLocaleString(),
        video: record.video ?? existing.video ?? '',
        log: record.log ?? existing.log ?? '',
        zip: record.zip ?? existing.zip ?? '',
        summary: record.summary ?? existing.summary ?? '',
        swingCsv: record.swingCsv ?? existing.swingCsv ?? ''
      };
      const next = [payload, ...prev.filter((item) => String(item.id) !== id)].slice(0, 50);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const updateHistoryStatus = useCallback((targetTaskId, nextStatus) => {
    upsertHistoryRecord({ id: targetTaskId, status: nextStatus, time: new Date().toLocaleString() });
  }, [upsertHistoryRecord]);

  useEffect(() => {
    if (!taskId) return;

    let cancelled = false;
    let pollingTimer = null;

    const fetchTaskStatus = async () => {
      try {
        const res = await axios.get(`/task/${taskId}/status`);
        if (cancelled) return;
        const data = res.data || {};

        if (data.status === 'processing') {
          setStatus('processing');
          upsertHistoryRecord({
            id: taskId,
            name: data.filename || selectedFileNameRef.current || `任务${taskId}`,
            status: 'processing',
            video: toAbsoluteUrl(data.video_url || ''),
            log: toAbsoluteUrl(data.log_url || ''),
            zip: toAbsoluteUrl(data.sub_videos_url || ''),
            summary: toAbsoluteUrl(data.summary_url || ''),
            swingCsv: toAbsoluteUrl(data.swing_csv_url || ''),
          });
        } else if (data.status === 'completed') {
          setStatus('completed');
          upsertHistoryRecord({
            id: taskId,
            name: data.filename || selectedFileNameRef.current || `任务${taskId}`,
            status: 'completed',
            video: toAbsoluteUrl(data.video_url || ''),
            log: toAbsoluteUrl(data.log_url || ''),
            zip: toAbsoluteUrl(data.sub_videos_url || ''),
            summary: toAbsoluteUrl(data.summary_url || ''),
            swingCsv: toAbsoluteUrl(data.swing_csv_url || ''),
          });
        } else if (data.status === 'paused') {
          setStatus('paused');
          updateHistoryStatus(taskId, 'paused');
        } else if (data.status === 'cancelled') {
          setStatus('cancelled');
          sessionStorage.removeItem(ACTIVE_TASK_KEY);
          sessionStorage.removeItem(ACTIVE_FILE_KEY);
          updateHistoryStatus(taskId, 'cancelled');
        } else if (data.status === 'failed') {
          setStatus('idle');
          if (data.error_message) {
            setError(data.error_message);
          }
          updateHistoryStatus(taskId, 'failed');
        }

        if (data.current_frame || data.total_frames) {
          setCurrentStats({
            frame: data.current_frame || 0,
            active_tracks: data.active_tracks || 0,
            tracks: data.tracks || []
          });
          setTotalFrames(data.total_frames || 0);
        }

        if (data.latest_frame) {
          setCurrentFrame(`data:image/jpeg;base64,${data.latest_frame}`);
        }

        if (data.video_url) {
          setVideoUrl(toAbsoluteUrl(data.video_url));
        }
        if (data.log_url) {
          setLogUrl(toAbsoluteUrl(data.log_url));
        }
        if (data.sub_videos_url) {
          setSubVideosZipUrl(toAbsoluteUrl(data.sub_videos_url));
        }
        if (data.summary_url) {
          setSummaryUrl(toAbsoluteUrl(data.summary_url));
        }
        if (data.swing_diag_url) {
          setSwingDiagUrl(toAbsoluteUrl(data.swing_diag_url));
        }
        if (data.swing_events_url) {
          setSwingEventsUrl(toAbsoluteUrl(data.swing_events_url));
        }
        if (data.swing_csv_url) {
          setSwingCsvUrl(toAbsoluteUrl(data.swing_csv_url));
        }
        if (Array.isArray(data.id_pose_summary)) {
          setIdPoseSummary(data.id_pose_summary);
        }

        if (data.status === 'completed') {
          sessionStorage.removeItem(ACTIVE_TASK_KEY);
          sessionStorage.removeItem(ACTIVE_FILE_KEY);
          try {
            const subRes = await axios.get(`/task/${taskId}/sub_videos`);
            if (cancelled) return;
            const videos = (subRes.data.videos || []).map((item) => ({
              ...item,
              url: toAbsoluteUrl(item.url)
            }));
            setSubVideos(videos);
            if (!selectedSubVideoRef.current && videos.length > 0) {
              setSelectedSubVideo(videos[0].url);
            }
          } catch (e) {
            console.error(e);
          }
        }
      } catch (e) {
        if (!cancelled) {
          console.error(e);
        }
      }
    };

    fetchTaskStatus();
    pollingTimer = window.setInterval(fetchTaskStatus, 1500);

    return () => {
      cancelled = true;
      if (pollingTimer) {
        window.clearInterval(pollingTimer);
      }
    };
  }, [taskId, updateHistoryStatus, upsertHistoryRecord]);

  const progress = useMemo(() => formatPercent(currentStats?.frame, totalFrames), [currentStats, totalFrames]);

  const resetCurrentOutput = () => {
    setCurrentFrame(null);
    setProcessingHint('请上传视频后开始识别。');
    setCurrentStats(null);
    setTotalFrames(0);
    setVideoUrl('');
    setLogUrl('');
    setSubVideosZipUrl('');
    setSummaryUrl('');
    setSwingCsvUrl('');
    setSwingEventsUrl('');
    setSwingDiagUrl('');
    setSubVideos([]);
    setSelectedSubVideo('');
    setIdPoseSummary([]);
    setError('');
  };

  const openResultUrl = (url) => {
    if (!url) return;
    window.open(toAbsoluteUrl(url), '_blank');
  };

  const loadDiskTasks = async () => {
    try {
      const { data } = await axios.get('/tasks');
      setDiskTasks(data.tasks || []);
      setDiskTotalMb(data.total_mb ?? 0);
      setDiskUploadMb(data.upload_total_mb ?? 0);
    } catch (e) {
      /* 后端未就绪时忽略 */
    }
  };

  const deleteDiskTask = async (taskId, withUpload) => {
    const what = withUpload ? '结果文件与上传原视频' : '结果文件';
    if (!window.confirm(`确定删除任务 ${taskId} 的${what}？此操作不可恢复。`)) return;
    try {
      const { data } = await axios.delete(`/task/${taskId}?with_upload=${withUpload ? 1 : 0}`);
      await loadDiskTasks();
      window.alert(`已删除 ${data.removed} 个文件，释放 ${data.freed_mb} MB`);
    } catch (e) {
      window.alert('删除失败：' + (e?.response?.data?.error || e.message));
    }
  };

  const clearDiskTasks = async (withUpload) => {
    const what = withUpload ? '全部结果与上传原视频' : '全部结果文件';
    if (!window.confirm(`确定清空${what}？此操作不可恢复。`)) return;
    try {
      const { data } = await axios.post('/tasks/clear', { with_upload: withUpload, keep_latest: 0 });
      await loadDiskTasks();
      window.alert(`已清理 ${data.tasks_deleted} 个任务、${data.removed} 个文件，释放 ${data.freed_mb} MB`);
    } catch (e) {
      window.alert('清理失败：' + (e?.response?.data?.error || e.message));
    }
  };

  useEffect(() => {
    loadDiskTasks();
  }, []);

  const resolveHistoryResourceUrl = async (item, resourceKey) => {
    if (item?.[resourceKey]) {
      return toAbsoluteUrl(item[resourceKey]);
    }
    if (!item?.id) {
      return '';
    }
    try {
      const res = await axios.get(`/task/${item.id}/status`);
      const data = res.data || {};
      const recovered = {
        id: String(item.id),
        name: item.name,
        status: data.status || item.status,
        video: toAbsoluteUrl(data.video_url || item.video || ''),
        log: toAbsoluteUrl(data.log_url || item.log || ''),
        zip: toAbsoluteUrl(data.sub_videos_url || item.zip || ''),
        summary: toAbsoluteUrl(data.summary_url || item.summary || ''),
        swingCsv: toAbsoluteUrl(data.swing_csv_url || item.swingCsv || ''),
      };
      upsertHistoryRecord(recovered);
      return recovered[resourceKey] || '';
    } catch (e) {
      console.error(e);
      return '';
    }
  };

  const handleHistoryOpen = async (item, resourceKey) => {
    const url = await resolveHistoryResourceUrl(item, resourceKey);
    if (!url) {
      setError('该历史任务暂无可访问结果，请重新处理该视频后再试。');
      return;
    }
    window.open(url, '_blank');
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setSelectedFileName(file.name);
    sessionStorage.setItem(ACTIVE_FILE_KEY, file.name);
    setStatus('idle');
    setProcessingHint('请上传视频后开始识别。');
    resetCurrentOutput();
  };

  const startProcess = async () => {
    if (!selectedFile) {
      setError('请先选择视频');
      return;
    }
    setError('');
    if (selectedFile && selectedFile.size > 500 * 1024 * 1024) {
      setStatus('idle');
      setError('视频文件过大（单个文件不能超过 500MB），请压缩或拆分后再上传。');
      return;
    }
    setStatus('uploading');

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const res = await axios.post('/upload', formData);
      const newTaskId = String(res.data.task_id);
      setTaskId(newTaskId);
      setStatus('processing');
      setProcessingHint('任务已提交，正在等待后端生成实时帧...');
      sessionStorage.setItem(ACTIVE_TASK_KEY, newTaskId);
      upsertHistoryRecord({
        id: newTaskId,
        name: selectedFile?.name || selectedFileName || `任务${newTaskId}`,
        status: 'processing'
      });
      socketRef.current?.emit('join', { task_id: newTaskId });
    } catch (e) {
      console.error(e);
      setStatus('idle');
      const httpStatus = e?.response?.status;
      const serverMsg = e?.response?.data?.error || e?.response?.data?.message || '';
      let detail = '';
      if (httpStatus === 413) detail = '视频文件过大（单个文件不能超过 500MB），请压缩或拆分后再上传';
      else if (httpStatus === 400) detail = serverMsg || '文件校验未通过（格式不支持或文件名异常）';
      else if (serverMsg) detail = serverMsg;
      setError(detail || '上传失败，请确认后端服务运行中；若仍失败请打开浏览器控制台(F12)查看具体报错。');
    }
  };

  const handleTaskControl = async (action) => {
    if (!taskId) return;
    try {
      const res = await axios.post(`/task/${taskId}/${action}`);
      const nextStatus = res.data?.status;
      if (nextStatus === 'paused') setStatus('paused');
      if (nextStatus === 'processing') setStatus('processing');
      if (nextStatus === 'cancelled') {
        setStatus('cancelled');
        sessionStorage.removeItem(ACTIVE_TASK_KEY);
        sessionStorage.removeItem(ACTIVE_FILE_KEY);
      }
      updateHistoryStatus(taskId, nextStatus || action);
    } catch (e) {
      console.error(e);
      setError('任务控制失败，请稍后重试');
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-grid" />
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      <header className="topbar glass-card">
        <div className="brand">
          <div className="brand-mark">🔬</div>
          <div>
            <h1>线虫运动识别</h1>
            <p className="subtitle">上传视频，实时查看识别画面、摆动次数与子视频</p>
          </div>
        </div>
        <div className="status-line">
          <div className={`socket-pill socket-${socketStatus}`}>
            Socket: {socketStatus}
          </div>
          {socketError && <div className="error-text">{socketError}</div>}
        </div>
      </header>

      <main className="layout">
        <aside className="panel glass-card left-panel">
          <h2>输入区</h2>
          <label className="upload-box" htmlFor="file-input">
            <input
              id="file-input"
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              disabled={status === 'uploading' || status === 'processing'}
            />
            <span className="upload-icon">📁</span>
            <span className="upload-title">点击选择线虫视频</span>
            <span className="upload-hint">支持 mp4 / avi / mov</span>
          </label>

          <div className="file-meta">
            <div className="row">
              <span className="key">文件</span>
              <span className="val">{selectedFile?.name || selectedFileName || '未选择'}</span>
            </div>
            <div className="row">
              <span className="key">类型</span>
              <span className="val">{getFileKind(selectedFile) || '未知'}</span>
            </div>
            <div className="row">
              <span className="key">状态</span>
              <span className="val">{status}</span>
            </div>
            <div className="row">
              <span className="key">任务 ID</span>
              <span className="val mono">{taskId || '无'}</span>
            </div>
          </div>

          <button
            type="button"
            className="glass-btn launch-btn"
            onClick={startProcess}
            disabled={!selectedFile || status === 'uploading' || status === 'processing' || status === 'paused'}
          >
            {status === 'uploading' ? '上传中...' : status === 'processing' ? '识别中...' : '启动识别'}
          </button>

          <div className="task-control-row">
            <button
              type="button"
              className="glass-btn"
              onClick={() => handleTaskControl('pause')}
              disabled={!taskId || status !== 'processing'}
            >暂停处理</button>
            <button
              type="button"
              className="glass-btn"
              onClick={() => handleTaskControl('resume')}
              disabled={!taskId || status !== 'paused'}
            >继续处理</button>
            <button
              type="button"
              className="glass-btn"
              onClick={() => handleTaskControl('cancel')}
              disabled={!taskId || ['completed', 'cancelled', 'idle'].includes(status)}
            >取消任务</button>
          </div>

          {error && <div className="error-text">{error}</div>}

          <div className="download-row">
            <button type="button" className="glass-btn" onClick={() => openResultUrl(videoUrl)} disabled={!videoUrl}>下载结果视频</button>
            <button type="button" className="glass-btn" onClick={() => openResultUrl(logUrl)} disabled={!logUrl}>下载日志</button>
            <button type="button" className="glass-btn" onClick={() => openResultUrl(subVideosZipUrl)} disabled={!subVideosZipUrl}>下载子视频ZIP</button>
            <button type="button" className="glass-btn" onClick={() => openResultUrl(summaryUrl)} disabled={!summaryUrl}>下载ID姿态汇总</button>
            <button type="button" className="glass-btn" onClick={() => openResultUrl(swingCsvUrl)} disabled={!swingCsvUrl}>下载摆动汇总CSV</button>
            <button type="button" className="glass-btn" onClick={() => openResultUrl(swingEventsUrl)} disabled={!swingEventsUrl}>下载摆动事件CSV</button>
            {swingDiagUrl && (
              <button type="button" className="glass-btn" onClick={() => openResultUrl(swingDiagUrl)}>下载摆动诊断CSV</button>
            )}
          </div>
        </aside>

        <section className="panel glass-card center-panel">
          <h2>实时处理区</h2>
          <div className="stream-wrap">
            {currentFrame ? (
              <img src={currentFrame} alt="实时处理画面" className="stream-frame" />
            ) : (
              <div className="stream-placeholder">{processingHint}</div>
            )}
          </div>

          <div className="progress-wrap">
            <div className="progress-text">进度 {currentStats?.frame || 0} / {totalFrames || 0}</div>
            <div className="progress-track">
              <div className="progress-value" style={{ width: `${progress}%` }} />
            </div>
          </div>

          <div className="stats-grid">
            <div className="stat-item">
              <span>活跃目标</span>
              <strong>{currentStats?.active_tracks ?? 0}<em>只</em></strong>
            </div>
            <div className="stat-item">
              <span>当前帧</span>
              <strong>{currentStats?.frame ?? 0}<em>帧</em></strong>
            </div>
            <div className="stat-item">
              <span>总帧数</span>
              <strong>{totalFrames || 0}<em>帧</em></strong>
            </div>
          </div>

          <section className="id-pose-section">
            <div className="sub-video-header">
              <h3>当前帧 ID 与姿态计数</h3>
            </div>
            {(currentStats?.tracks || []).length === 0 ? (
              <div className="history-empty">当前帧暂无可显示的线虫计数</div>
            ) : (
              <div className="id-pose-table-wrap">
                <table className="id-pose-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>摆动(单方向/次)</th>
                      <th>完整周期(次)</th>
                      <th>实时摆动(Hz)</th>
                      <th>平均摆动(Hz)</th>
                      <th>次/分钟</th>
                      <th>状态</th>
                      <th>摆动活性</th>
                      <th>运动活性</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(currentStats?.tracks || []).map((item) => (
                      <tr key={`live-${item.track_id}`}>
                        <td>{item.track_id}</td>
                        <td>{item.swing_count ?? item.swings ?? item.cycles ?? item.posture_count ?? 0}</td>
                        <td>{item.cycle_count ?? 0}</td>
                        <td>{item.is_stale ? '--' : Number(item.live_freq ?? item.realtime_freq ?? item.freq ?? 0).toFixed(2)}</td>
                        <td>{Number(item.avg_freq ?? item.average_frequency ?? item.freq ?? 0).toFixed(2)}</td>
                        <td>{Number(item.bends_per_min ?? 0).toFixed(0)}</td>
                        <td>
                          <span className={`badge ${item.is_stale ? 'badge-ghost' : (item.is_ghost ? 'badge-ghost' : 'badge-ok')}`}>
                            {item.is_stale
                              ? `未观测 ${Math.round(Number(item.unobserved_seconds ?? 0))}s`
                              : (item.is_ghost ? `残留 丢${item.lost_frames ?? 0}帧` : '正常')}
                          </span>
                        </td>
                        <td>{item.is_stale ? '--' : `${Math.round(Number(item.swing_activity ?? 0) * 100)}%`}</td>
                        <td>{item.is_stale ? '--' : `${Math.round(Number(item.activity_score ?? item.motion_activity ?? 0) * 100)}%`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="id-pose-section">
            <div className="sub-video-header">
              <h3>最终输出 ID 与姿态计数</h3>
            </div>
            {idPoseSummary.length === 0 ? (
              <div className="history-empty">任务完成后会显示最终汇总</div>
            ) : (
              <div className="id-pose-table-wrap">
                <table className="id-pose-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>摆动(单方向/次)</th>
                      <th>完整周期(次)</th>
                      <th>实时摆动(Hz)</th>
                      <th>平均摆动(Hz)</th>
                      <th>次/分钟</th>
                      <th>状态</th>
                      <th>摆动活性</th>
                      <th>运动活性</th>
                    </tr>
                  </thead>
                  <tbody>
                    {idPoseSummary.map((item) => (
                      <tr key={`summary-${item.track_id}`}>
                        <td>{item.track_id}</td>
                        <td>{item.swing_count ?? item.swings ?? item.cycles ?? item.posture_count ?? 0}</td>
                        <td>{item.cycle_count ?? 0}</td>
                        <td>{item.is_stale ? '--' : Number(item.live_freq ?? item.realtime_freq ?? item.freq ?? 0).toFixed(2)}</td>
                        <td>{Number(item.avg_freq ?? item.average_frequency ?? item.freq ?? 0).toFixed(2)}</td>
                        <td>{Number(item.bends_per_min ?? 0).toFixed(0)}</td>
                        <td>
                          <span className={`badge ${item.is_stale ? 'badge-ghost' : (item.is_ghost ? 'badge-ghost' : 'badge-ok')}`}>
                            {item.is_stale
                              ? `未观测 ${Math.round(Number(item.unobserved_seconds ?? 0))}s`
                              : (item.is_ghost ? `残留 丢${item.lost_frames ?? 0}帧` : '正常')}
                          </span>
                        </td>
                        <td>{item.is_stale ? '--' : `${Math.round(Number(item.swing_activity ?? 0) * 100)}%`}</td>
                        <td>{item.is_stale ? '--' : `${Math.round(Number(item.activity_score ?? item.motion_activity ?? 0) * 100)}%`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {videoUrl && (
            <GlassVideoPlayer title="处理后主视频" src={videoUrl} downloadName="main_result" />
          )}

          {subVideos.length > 0 && (
            <section className="sub-video-section">
              <div className="sub-video-header">
                <h3>线虫子视频</h3>
                <select
                  value={selectedSubVideo}
                  onChange={(e) => setSelectedSubVideo(e.target.value)}
                  className="glass-select"
                >
                  {subVideos.map((item) => (
                    <option key={item.filename} value={item.url}>
                      {item.filename}
                    </option>
                  ))}
                </select>
              </div>
              {selectedSubVideo && (
                <GlassVideoPlayer title="当前线虫局部子视频" src={selectedSubVideo} downloadName="worm_sub" />
              )}
            </section>
          )}
        </section>

        <aside className="panel glass-card history-panel">
          <h2>历史</h2>
          {historyList.length === 0 && <div className="history-empty">暂无历史记录</div>}
          <div className="history-list">
            {historyList.map((item) => (
              <article key={item.id + item.time} className="history-item">
                <div className="history-title">{item.name}</div>
                <div className="history-time">{item.time}</div>
                <div className="history-time">状态: {item.status || 'completed'}</div>
                <div className="history-actions">
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'video')}>查看</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'log')}>日志</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'zip')}>子视频</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'summary')}>汇总</button>
                  <button type="button" className="glass-btn mini" onClick={() => handleHistoryOpen(item, 'swingCsv')}>摆动CSV</button>
                </div>
              </article>
            ))}
          </div>

          <section className="disk-section">
            <div className="sub-video-header">
              <h3>磁盘结果：{diskTasks.length} 个任务 / {diskTotalMb} MB</h3>
            </div>
            <div className="history-actions">
              <button type="button" className="glass-btn mini" onClick={() => clearDiskTasks(false)} disabled={diskTasks.length === 0}>清空结果</button>
              <button type="button" className="glass-btn mini" onClick={() => clearDiskTasks(true)} disabled={diskTasks.length === 0}>清空结果+原视频</button>
              <button type="button" className="glass-btn mini" onClick={loadDiskTasks}>刷新</button>
            </div>
            {diskTasks.length === 0 ? (
              <div className="history-empty">暂无磁盘结果文件</div>
            ) : (
              <div className="history-list">
                {diskTasks.map((item) => (
                  <article key={`disk-${item.task_id}`} className="history-item">
                    <div className="history-title">{item.task_id}</div>
                    <div className="history-time">{item.time} · {item.files} 个文件 · {item.mb} MB{item.upload ? ` · 原视频 ${item.upload_mb} MB` : ''}</div>
                    <div className="history-time">状态: {item.running ? '处理中' : '已完成'}</div>
                    <div className="history-actions">
                      <button type="button" className="glass-btn mini" onClick={() => deleteDiskTask(item.task_id, false)} disabled={item.running}>删除结果</button>
                      <button type="button" className="glass-btn mini" onClick={() => deleteDiskTask(item.task_id, true)} disabled={item.running || !item.upload}>删除结果+原视频</button>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;