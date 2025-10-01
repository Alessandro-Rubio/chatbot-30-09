import React, { useState, useEffect } from 'react';
import { fileAPI, ragAPI } from '../services/api';

const FileManagement = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [ragStatus, setRagStatus] = useState({});

  useEffect(() => {
    loadFiles();
    loadRAGStatus();
  }, []);

  const loadFiles = async () => {
    try {
      const response = await fileAPI.listFiles();
      setFiles(response.files);
    } catch (error) {
      console.error('Error loading files:', error);
    }
  };

  const loadRAGStatus = async () => {
    try {
      const status = await ragAPI.getStatus();
      setRagStatus(status);
    } catch (error) {
      console.error('Error loading RAG status:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    try {
      const result = await fileAPI.uploadFile(formData);
      
      if (result.status === 'duplicate') {
        alert('El archivo ya existe en el sistema');
      } else {
        await loadFiles();
        await loadRAGStatus();
        alert('Archivo subido exitosamente. El sistema RAG se está actualizando.');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error al subir el archivo');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleDeleteFile = async (filename) => {
    if (window.confirm(`¿Eliminar el archivo ${filename}?`)) {
      try {
        await fileAPI.deleteFile(filename);
        await loadFiles();
        await loadRAGStatus();
        alert('Archivo eliminado exitosamente');
      } catch (error) {
        console.error('Error deleting file:', error);
        alert('Error al eliminar el archivo');
      }
    }
  };

  const handleReinitializeRAG = async () => {
    try {
      await ragAPI.reinitialize();
      alert('Sistema RAG se está reinicializando...');
      setTimeout(loadRAGStatus, 3000); // Verificar estado después de 3 segundos
    } catch (error) {
      console.error('Error reinitializing RAG:', error);
    }
  };

  return (
    <div className="file-management">
      <div className="rag-status-card">
        <h4>Estado del Sistema RAG</h4>
        <div className="status-indicators">
          <span className={`status ${ragStatus.rag_initialized ? 'active' : 'inactive'}`}>
            {ragStatus.rag_initialized ? '🟢 RAG Activo' : '🔴 RAG Inactivo'}
          </span>
          <span className="file-count">
            📁 {ragStatus.file_count || 0} archivos
          </span>
          <button 
            onClick={handleReinitializeRAG}
            className="reinit-btn"
          >
            🔄 Reinicializar RAG
          </button>
        </div>
      </div>

      <div className="upload-section">
        <h4>Subir Nuevo Archivo</h4>
        <div className="upload-area">
          <input
            type="file"
            onChange={handleFileUpload}
            disabled={uploading}
            accept=".pdf,.txt,.docx,.doc"
          />
          {uploading && <div className="uploading">Subiendo...</div>}
        </div>
        <small>Formatos soportados: PDF, TXT, DOCX, DOC</small>
      </div>

      <div className="file-list-section">
        <h4>Archivos en el Sistema</h4>
        {files.length === 0 ? (
          <p className="no-files">No hay archivos cargados</p>
        ) : (
          <div className="file-grid">
            {files.map((file, index) => (
              <div key={index} className="file-card">
                <div className="file-info">
                  <div className="filename">{file.filename}</div>
                  <div className="file-meta">
                    {file.metadata && (
                      <>
                        <span>Tamaño: {(file.metadata.size / 1024).toFixed(1)} KB</span>
                        <span>Tipo: {file.metadata.file_type}</span>
                      </>
                    )}
                  </div>
                </div>
                <button 
                  onClick={() => handleDeleteFile(file.filename)}
                  className="delete-btn"
                >
                  🗑️ Eliminar
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default FileManagement;